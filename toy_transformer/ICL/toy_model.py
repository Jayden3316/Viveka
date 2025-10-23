
import os

import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib.figure import Figure

from jaxtyping import Float
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.train import HookedTransformerTrainConfig
from metrics import MetricsConfig, generate_prefix_matching_data, compute_metrics
from mealymarkov import MarkovMealyModel
from typing import List, Optional, Tuple,Literal
import scipy
from sklearn.linear_model import LinearRegression

def run_activation_to_beliefs_regression(activations, ground_truth_beliefs):

    # make sure the first two dimensions are the same
    assert activations.shape[0] == ground_truth_beliefs.shape[0]
    assert activations.shape[1] == ground_truth_beliefs.shape[1]

    # flatten the activations
    batch_size, n_ctx, d_model = activations.shape
    belief_dim = ground_truth_beliefs.shape[-1]
    activations_flattened = activations.reshape(-1, d_model)  # [batch * n_ctx, d_model]
    ground_truth_beliefs_flattened = ground_truth_beliefs.reshape(
        -1, belief_dim
    )  # [batch * n_ctx, belief_dim]

    # run the regression
    regression = LinearRegression()
    regression.fit(activations_flattened, ground_truth_beliefs_flattened)

    # get the belief predictions
    belief_predictions = regression.predict(
        activations_flattened
    )  # [batch * n_ctx, belief_dim]
    belief_predictions = belief_predictions.reshape(batch_size, n_ctx, belief_dim)

    return regression, belief_predictions


def _project_to_simplex(points:np.ndarray):
    """Project points onto the 2-simplex (equilateral triangle in 2D)."""
    x = points[:, 1] + 0.5 * points[:, 2]
    y = (np.sqrt(3) / 2) * points[:, 2]
    return x, y

# Combine aggregated channels into RGB images
def _combine_channels_to_rgb(agg_r, agg_g, agg_b, px:int):
    img_r = tf.shade(agg_r, cmap=['black', 'red'], how='linear')
    img_g = tf.shade(agg_g, cmap=['black', 'green'], how='linear')
    img_b = tf.shade(agg_b, cmap=['black', 'blue'], how='linear')

    img_r = tf.spread(img_r, px=px, shape='circle')
    img_g = tf.spread(img_g, px=px, shape='circle')
    img_b = tf.spread(img_b, px=px, shape='circle')

    # Combine using numpyß
    r_array = np.array(img_r.to_pil()).astype(np.float64)
    g_array = np.array(img_g.to_pil()).astype(np.float64)
    b_array = np.array(img_b.to_pil()).astype(np.float64)

    
    # Stack arrays into an RGB image (ignoring alpha channel for simplicity)
    rgb_image = np.stack([r_array[:,:,0], g_array[:,:,1], b_array[:,:,2]], axis=-1)
    
    return Image.fromarray(np.uint8(rgb_image))

# TODO: I changed up the code for this to something which makes sense to me (creating panda dataframes from ground truth and predicted simplex. Check to see if this is what should actually be done)
def plot_ground_truth_and_evaluated_2d_simplex(
    ground_truth_tensor: np.ndarray, 
    predicted_beliefs: np.ndarray, 
    plot_triangles: bool,
    facecolor: Literal['black', 'white'],
    px: int
) -> Figure:
    # Projection and DataFrame preparation
    ground_truth_tensor=ground_truth_tensor.reshape(-1,ground_truth_tensor.shape[-1])
    print(ground_truth_tensor.shape)
    bs_x, bs_y = _project_to_simplex(np.array(ground_truth_tensor))
    bs_x, bs_y = np.ravel(bs_x), np.ravel(bs_y)
    print("bs_x shape:", np.shape(bs_x))
    print("bs_y shape:", np.shape(bs_y))
    print("ground_truth_tensor shape:", np.shape(ground_truth_tensor))
   #ground_truth_tensor=ground_truth_tensor.reshape(-1,ground_truth_tensor.shape[-1])
    ground_truth_data_frame = pd.DataFrame({'x': bs_x, 'y': bs_y, 'r': ground_truth_tensor[:, 0], 'g': ground_truth_tensor[:, 1], 'b': ground_truth_tensor[:, 2]})
    predicted_beliefs=predicted_beliefs.reshape(-1,predicted_beliefs.shape[-1])
    print(predicted_beliefs.shape)
    pb_x, pb_y = _project_to_simplex(np.array(predicted_beliefs))
    pb_x, pb_y = np.ravel(pb_x), np.ravel(pb_y)
    print("pb_x shape:", np.shape(pb_x))
    print("pb_y shape:", np.shape(pb_y))
    print("predicted_belief shape:", np.shape(predicted_beliefs))
    predicted_belief_vector_data_frame = pd.DataFrame({'x': pb_x, 'y': pb_y, 'r': ground_truth_tensor[:, 0], 'g': ground_truth_tensor[:, 1], 'b': ground_truth_tensor[:, 2]})

    # Create canvas
    canvas = ds.Canvas(plot_width=1000, plot_height=1000, x_range=(-0.1, 1.1), y_range=(-0.1, np.sqrt(3)/2 + 0.1))
    
    # Aggregate each RGB channel separately for ground truth and predicted beliefs
    colours = ['r', 'g', 'b']
    ground_truth_aggregated = {color: canvas.points(ground_truth_data_frame, 'x', 'y', ds.mean(color)) for color in colours}
    predicted_belief_vector_aggregated = {color: canvas.points(predicted_belief_vector_data_frame, 'x', 'y', ds.mean(color)) for color in colours}

    img_gt = _combine_channels_to_rgb(ground_truth_aggregated['r'], ground_truth_aggregated['g'], ground_truth_aggregated['b'], px=2*px)
    img_pb = _combine_channels_to_rgb(predicted_belief_vector_aggregated['r'], predicted_belief_vector_aggregated['g'], predicted_belief_vector_aggregated['b'], px=px)

    # Visualization with Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, facecolor=facecolor)
    for ax in axs:
        ax.tick_params(axis='x', colors=facecolor)  
        ax.tick_params(axis='y', colors=facecolor)  
        ax.xaxis.label.set_color(facecolor)  
        ax.yaxis.label.set_color(facecolor)  
        ax.title.set_color(facecolor)
    axs[0].imshow(img_gt)
    axs[1].imshow(img_pb)
    
    axs[0].axis('off')
    axs[1].axis('off')
    title_y_position = -0.1  # Adjust this value to move the title up or down relative to the axes
    fig.text(0.5, title_y_position, 'Ground Truth', ha='center', va='top', transform=axs[0].transAxes, color='white', fontsize=15)  # Changed 'black' to 'white'
    fig.text(0.5, title_y_position, 'Residual Stream', ha='center', va='top', transform=axs[1].transAxes, color='white', fontsize=15)  # Changed 'black' to 'white'

    if plot_triangles:
        for ax in axs:
            ax.plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0], 'white', lw=2)  # Changed 'black' to 'white'

    return fig    

class ParallelMarkovGeneratorGPU:
    """
    Generates Markov sequences in a fully parallelized batch on a GPU.
    """
    def __init__(self, n_states: int,n_gen:int, gen_len:int, d_vocab: int, T_list: list[np.ndarray], 
                 eta0: Optional[np.ndarray] = None, device: str = 'cuda', seed:int=42 ):
        
        if not torch.cuda.is_available() and device == 'cuda':
            print("CUDA not available, falling back to CPU. Performance will be slower.")
            device = 'cpu'
        self.device = torch.device(device)
        self.n_gen=n_gen
        self.gen_len=gen_len
        self.seed=seed
        self.n_states = n_states
        self.d_vocab = d_vocab
        
        # Stack T_list into a single tensor of shape (V, n, n) and move to GPU
        self.T_stack = torch.tensor(np.array(T_list), dtype=torch.float32).to(self.device)

        # Prepare initial state eta0
        if eta0 is None:
            eta0 = np.full((self.n_states,), 1.0 / self.n_states)
        self.eta0 = torch.tensor(eta0, dtype=torch.float32).to(self.device)
        
        self.data=[]
        self.states=[]
        self.generate_batch()
    def generate_batch(self):
        """
        Generates a batch of n_gen sequences, each of length gen_len.
        
        Returns:
            torch.Tensor: A tensor of token indices with shape (n_gen, gen_len).
        """
        # Initialize the states for the entire batch
        # Shape: (n_gen, n_states)
        n_gen, gen_len = self.n_gen, self.gen_len
        eta_batch = self.eta0.expand(n_gen, -1)
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed)

        # List to store the generated tokens for each step
        generated_tokens = []
        generated_probs = []
        generated_states=[]
        generated_states.append(eta_batch.cpu())

        for _ in range(gen_len):
            # --- 1. Calculate token probabilities for the entire batch ---
            # We use einsum for a clear and efficient batched matrix multiplication.
            # 'bi,vij->bvj' means: for each item 'b' in the batch and each vocab item 'v',
            # multiply state (bi) with matrix (vij) to get the next unnormalized state (bvj).
            # eta_batch shape:      (n_gen, n_states)
            # self.T_stack shape: (d_vocab, n_states, n_states)
            unnorm_next_eta_batch = torch.einsum('bi,vij->bvj', eta_batch, self.T_stack)
            
            # Sum over the last dimension (j) to get token probabilities
            # Shape: (n_gen, d_vocab)
            p_batch = unnorm_next_eta_batch.sum(dim=-1)
            #p_batch=p_batch/(p_batch.sum(dim=-1,keepdim=True)) # Normalize to get probabilities, add epsilon for stability
            
            # --- 2. Sample the next token for the entire batch ---
            # torch.multinomial samples from the distributions in p_batch
            # Shape: (n_gen, 1)
            next_token_batch = torch.multinomial(p_batch, num_samples=1,generator=g)
            generated_tokens.append(next_token_batch.cpu())
            generated_probs.append(p_batch.view(n_gen, 1, self.d_vocab).cpu())
            
            # --- 3. Evolve the state for the entire batch ---
            # Get the specific unnormalized next state that corresponds to the chosen token
            # This is a highly efficient way to select the right T^k for each item in the batch
            # Shape: (n_gen, n_states)
            next_eta_numer = torch.gather(
                unnorm_next_eta_batch, 
                1, 
                next_token_batch.unsqueeze(-1).expand(-1, -1, self.n_states)
            ).squeeze(1)

            # Normalize to get the next state distribution
            next_eta_denom = next_eta_numer.sum(dim=-1, keepdim=True)
            eta_batch = next_eta_numer / (next_eta_denom) # Add epsilon for stability
            del next_eta_denom, next_eta_numer, p_batch, next_token_batch, unnorm_next_eta_batch
            generated_states.append(eta_batch.cpu())

        token_tensor=torch.stack(generated_tokens).squeeze(-1).T 
        state_tensor=torch.stack(generated_states).permute(1,0,2)  

        self.data = [token_tensor[i].to(self.device) for i in range(self.n_gen)]
        self.states = [
        [arr for arr in seq.cpu().numpy().astype(float)]
        for seq in state_tensor]
    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx):
            return {"tokens": self.data[idx]}

    def to(self, device):
            if str(device) != str(self.device):
                self.data = [tensor.to(device) for tensor in self.data]
                self.device = torch.device(device)
            return self

    def get_stacked_data(self):
            return torch.stack(self.data)


class MarkovData(Dataset):

    def __init__(
        self,
        n_gen: int,
        gen_len: int,
        n_states: int,
        d_vocab: int,
        T_list: list[np.ndarray],
        eta0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        seed: int = 42,
        device:str="cpu"
    ):
        self.model = MarkovMealyModel(n_states, d_vocab, T_list, eta0, rng)
        self.d_vocab = self.model.V
        self.gen_len = gen_len
        self.device = device
        self.data = []
        self.states = []
        rng = rng or np.random.default_rng(seed)

        for i in range(n_gen):
            tokens, states = self.model.sample_sequence(
                max_new_tokens=gen_len,
                seed=rng.integers(2**32)
            )

            self.data.append(torch.tensor(tokens, dtype=torch.int64,device=device))
            self.states.append(states)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"tokens": self.data[idx]}
    def to(self, device):
        if device!=self.device:
            self.data = [tensor.to(device) for tensor in self.data]
            self.device = device
        return self        

    def get_stacked_data(self):
        return torch.stack(self.data)


class MergeMarkovDatasets(Dataset):
    """
    Merges two MarkovData objects and returns a new Dataset object.

    Mixing style is the manner in which the datasets should be merged:
    - "random": Generations from both the datasets are randomly mixed.
    - "alternate": The new dataset has generations alternating from both the datasets.
    - "stack": Generations of the second dataset are added after generations of the first dataset.

    Note that `mixing_style` may play an important role in training of the model.
    """

    def __init__(
        self, 
        dataset1: MarkovData, 
        dataset2: MarkovData, 
        mixing_style: Literal["random", "alternate", "stack"]
    ):
        self.model1 = dataset1.model
        self.model2 = dataset2.model

        assert dataset1.d_vocab == dataset2.d_vocab, "Vocabulary size for the datasets does not match"
        self.d_vocab = dataset1.d_vocab

        assert dataset1.gen_len == dataset2.gen_len, "Generation lengths for the datasets do not match"
        self.gen_len = dataset1.gen_len

        if dataset1.device != dataset2.device:
            dataset2.to(dataset1.device)
        self.device = dataset1.device

        data1 = list(zip(dataset1.data, dataset1.states))
        data2 = list(zip(dataset2.data, dataset2.states))

        if mixing_style == "random":
            merged = data1 + data2
            np.random.shuffle(merged)
        elif mixing_style == "alternate":
            assert len(data1) == len(data2), "Mixing style 'alternate' is valid only when the size of both datasets is same"
            merged = []
            for i in range(len(data1)):
                merged.append(data1[i])
                merged.append(data2[i])
        else:
            merged = data1 + data2

        self.data = [d for d, s in merged]
        self.states = [s for d, s in merged]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"tokens": self.data[idx]}

    def to(self,device):
        if device!=self.device:
            self.data = [tensor.to(device) for tensor in self.data]
            self.device=device
        return self

    def get_stacked_data(self):
        return torch.stack(self.data)    


def train(
    model: HookedTransformer,
    config: HookedTransformerTrainConfig,
    train_data: Dataset,
    val_data: Optional[Dataset] = None,
    eval_every: Optional[int] = None,
    metrics_config: Optional[MetricsConfig] = None,
    metrics_log_interval: int = 50
) -> HookedTransformer:
    """
    Helper function to train an HookedTransformer model on an autoregressive language modeling task.
    Slightly modified version of TransformerLens one with advanced metrics tracking integration.

    Args:
        model: The model to train
        config: The training configuration
        train_data: The dataset to train on
        val_data: The dataset to use for validation
        eval_every: Number of epochs after which to run the model on val_data
        metrics_tracker: AdvancedMetricsTracker instance for logging metrics

    Returns:
        The trained model
    """
    torch.manual_seed(config.seed)

    model.train()

    if config.wandb:
        if config.wandb_project_name is None:
            config.wandb_project_name = "easy-transformer"
        wandb.init(project=config.wandb_project_name, config=vars(config))

    # Set up optimizer
    if config.optimizer_name in ["Adam", "AdamW"]:
        if config.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=config.lr, 
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay if config.weight_decay is not None else 0.0,
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} not supported")

    # Set up learning rate scheduler
    scheduler = None
    if config.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
        )

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=len(val_data)) if val_data else None

    model = model.to(config.device)
    if metrics_config is not None:
        metrics_config.device=config.device

    global_step = 0

    for epoch in tqdm(range(1, config.num_epochs + 1)):
        samples = 0
        for step, batch in enumerate(train_dataloader):
            tokens = batch["tokens"].to(config.device)

            loss = model(tokens, return_type="loss")
            loss.backward()

            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            if config.warmup_steps > 0:
                assert scheduler is not None
                scheduler.step()
            optimizer.zero_grad()

            samples += tokens.shape[0]
            global_step += 1

            if config.wandb:
                wandb.log({"train_loss": loss.item(), "samples": samples, "epoch": epoch}, step=global_step)

            if metrics_config is not None and global_step % metrics_log_interval == 0:
                #try:
                model.eval()
                with torch.no_grad():
                    compute_metrics(model, metrics_config, global_step)
                model.train()
                    
                if global_step % (metrics_log_interval) == 0:
                    print(f"Metrics logged at step {global_step}")
                #except Exception as e:
                 #   print(f"Warning: Error in metrics tracking at step {global_step}: {e}")
            

        if config.print_every is not None and epoch % config.print_every == 0:
            print(f"Epoch {epoch} Samples {samples} Step {step} Training Loss {loss.item()}")

        if config.save_every is not None and epoch % config.save_every == 0 and config.save_dir is not None:
            torch.save(model.state_dict(), f"{config.save_dir}/model{epoch}.pt")

        if val_dataloader and eval_every is not None and epoch % eval_every == 0:
            for data in val_dataloader:
                model.eval()
                tokens = data["tokens"].to(config.device)
                with torch.no_grad():
                    loss = model(tokens, return_type="loss")
                if config.wandb:
                    wandb.log({"val_loss": loss.item(), "epoch": epoch}, step=global_step)
                print(f"Epoch {epoch} Validation Loss {loss.item()}")

    return model


def train_model(
    dataset: MarkovData | MergeMarkovDatasets,
    # Transformer Architecture
    n_layers: int = 4,
    d_model: int = 64,
    n_heads: int = 1,
    d_head: int = 8,
    attn_only: bool = False,
    d_mlp: int = 256,
    act_fn: Literal["relu", "gelu", "silu", "gelu_new", "solu_ln", "gelu_fast"] = "relu",
    normalization_type: Literal["LN", "LNPre", "RMS", "RMSPre"] | None = None,
    positional_embedding_type: Literal["standard", "rotary", "shortformer"] = "standard",
    default_prepend_bos: bool = True,
    # Training Hyperparameters
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-2,
    optimizer_name: Literal["Adam", "AdamW", "SGD"] = "SGD",
    wandb: bool = False,
    wandb_project_name: Optional[str] = None,
    # System + I/O
    device: str = "cpu",
    seed: int = 42,
    save_every: int = 1,
    save_dir: str = "./checkpoints",
    print_every: int = 1,
    eval_every: int = 1,
    val_frac: float = 0.2,
    # Metrics tracking
    metrics_config: Optional[MetricsConfig] = None,
    metrics_log_interval: int = 50,

    track_ngrams: bool = True,
    ngram_orders: list[int] = [1,2, 3],
    track_previous_token: bool = True,
    track_in_context: bool = True,
    track_composition: bool = True,
    track_prefix_matching: bool = True
) -> HookedTransformer:
    """
    Train a HookedTransformer on sequences generated from a Mealy Markov model.

    This function constructs a HookedTransformer model with the given architecture 
    and optimization hyperparameters, and trains it on sequences generated from a 
    custom Markov process dataset.

    Parameters
    ----------
    dataset : MarkovData or MergeMarkovDatasets
        Training dataset containing token sequences generated from a Mealy-Markov process.
    n_layers : int
        Number of transformer layers.
    d_model : int
        Dimension of the model embedding and hidden sizes.
    n_heads : int
        Number of attention heads.
    d_head : int
        Dimension per attention head.
    attn_only : bool
        Whether the transformer is attention-only, without any MLP blocks.
    d_mlp : int
        Dimension of the feedforward hidden layer.
    act_fn : {"relu", "gelu", "silu", "gelu_new", "solu_ln", "gelu_fast"}
        Activation function used in MLP layers.
    normalization_type : {"LN", "LNPre", "RMS", "RMSPre"}
        Normalization strategy applied in transformer layers. Defaults to no normalization
    positional_embedding_type : {"standard", "rotary", "shortformer"}
        Type of positional embeddings used in the model
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for optimization.
    optimizer_name : {"Adam", "AdamW", "SGD"}
        Optimizer to use.
    wandb : bool
        Whether to use wandb to log training
    wandb_project_name : str, optional
        Name for wandb project, defaults to "easy-transformer"
    device : str
        Device where the model will be trained (e.g., "cpu", "cuda").
    seed : int
        Random seed for reproducibility
    save_every : int
        Frequency (in epochs) to checkpoint the model.
    save_dir : str
        Directory where checkpoints will be saved.
    print_every : int
        Frequency (in epochs) to log training progress.
    eval_every : int
        Evaluate on a validation dataset
    val_frac : float
        Fraction of dataset to be used as validation dataset
    track_ngrams : bool
        Whether to track n-gram metrics during training
    ngram_orders : list[int]
        List of n-gram orders to track
    track_sets : list[str]
        List of dataset splits to track metrics for
    track_composition : bool
        Whether to track attention head composition scores
    track_previous_token : bool
        Whether to track previous token matching scores
    track_in_context : bool
        Whether to track in-context learning scores
    track_prefix_matching : bool
        Whether to track prefix matching scores

    Returns
    -------
    HookedTransformer
        The trained transformer model.
    """
    d_vocab = dataset.d_vocab
    n_ctx = dataset.gen_len +1

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_ctx=n_ctx,
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=d_mlp,
        act_fn=act_fn,
        d_vocab=d_vocab,
        attn_only=attn_only,
        normalization_type=normalization_type,
        device=device,
        positional_embedding_type=positional_embedding_type,
        seed=seed,
        default_prepend_bos=True,
    )

    model = HookedTransformer(cfg, move_to_device=True)

    # System + I/O
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_dir is not None:
        torch.save(cfg, f"{save_dir}/model_cfg.pt")

    # System + I/O
    train_cfg = HookedTransformerTrainConfig(
        num_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer_name=optimizer_name,
        wandb=wandb,
        wandb_project_name=wandb_project_name,
        device=device,
        seed=seed,
        save_every=save_every,
        save_dir=save_dir,
        print_every=print_every
    )

    # Initialize ADVANCED metrics tracker if any metrics tracking is requested
    if metrics_config is None and any([track_ngrams, track_composition, track_previous_token, 
                                     track_in_context, track_prefix_matching]):
        metrics_config = MetricsConfig(
            track_ngrams=track_ngrams,
            ngram_orders=ngram_orders,
            track_composition=track_composition,
            track_previous_token=track_previous_token,
            track_in_context=track_in_context,
            track_prefix_matching=track_prefix_matching,
            device=device
        )
        print("Created metrics config from individual parameters")

    # Train-val split
    if val_frac:
        train_size = int(len(dataset) * (1 - val_frac))
        indices = torch.randperm(len(dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_data = Subset(dataset, train_indices)
        val_data = Subset(dataset, val_indices)
        return train(model, train_cfg, train_data, val_data, eval_every, metrics_config)
    else:
        return train(model, train_cfg, dataset, metrics_config=metrics_config)


def finetune_model(
    model: HookedTransformer,
    dataset: MarkovData | MergeMarkovDatasets,
    n_epochs: int,
    batch_size: int = 64,
    lr: float = 1e-2,
    optimizer_name: Literal["Adam", "AdamW", "SGD"] = "SGD",
    wandb: bool = False,
    wandb_project_name: Optional[str] = None,
    device: str = "cpu",
    seed: int = 42,
    save_every: int = 1,
    save_dir: str = "./checkpoints",
    print_every: int = 1,
    eval_every: int = 1,
    val_frac: float = 0.2,
    # Advanced Metrics tracking
    track_ngrams: bool = True,
    ngram_orders: list[int] = [2, 3, 4],
    track_sets: list[str] = ["train", "val", "complete"],
    track_composition: bool = True,
    track_previous_token: bool = True,
    track_in_context: bool = True,
    track_prefix_matching: bool = True
) -> HookedTransformer:
    """
    Finetune a pretrained HookedTransformer on sequences generated from a Mealy Markov model

    Parameters
    ----------
    model : HookedTransformer
        A pre-trained model to finetune.
    dataset : MarkovData or MergeMarkovDatasets
        Training dataset containing token sequences generated from a Mealy-Markov process.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for optimization.
    optimizer_name : {"Adam", "AdamW", "SGD"}
        Optimizer to use.
    wandb : bool
        Whether to use wandb to log training
    wandb_project_name : str, optional
        Name for wandb project, defaults to "easy-transformer"
    device : str
        Device where the model will be trained (e.g., "cpu", "cuda").
    seed : int
        Random seed for reproducibility
    save_every : int
        Frequency (in epochs) to checkpoint the model.
    save_dir : str
        Directory where checkpoints will be saved.
    print_every : int
        Frequency (in epochs) to log training progress.
    eval_every : int
        Evaluate on a validation dataset
    val_frac : float
        Fraction of dataset to be used as validation dataset
    track_ngrams : bool
        Whether to track n-gram metrics during training
    ngram_orders : list[int]
        List of n-gram orders to track
    track_sets : list[str]
        List of dataset splits to track metrics for
    track_composition : bool
        Whether to track attention head composition scores
    track_previous_token : bool
        Whether to track previous token matching scores
    track_in_context : bool
        Whether to track in-context learning scores
    track_prefix_matching : bool
        Whether to track prefix matching scores

    Returns
    -------
    HookedTransformer
        The trained transformer model.
    """
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cfg = HookedTransformerTrainConfig(
        num_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer_name=optimizer_name,
        wandb=wandb,
        wandb_project_name=wandb_project_name,
        device=device,
        seed=seed,
        save_every=save_every,
        save_dir=save_dir,
        print_every=print_every
    )
    # Train-val split
    if val_frac:
        train_size = int(len(dataset) * (1 - val_frac))
        indices = torch.randperm(len(dataset))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_data = Subset(dataset, train_indices)
        val_data = Subset(dataset, val_indices)
        return train(model, cfg, train_data, val_data, eval_every, metrics_tracker)
    else:
        return train(model, cfg, dataset, metrics_tracker=metrics_tracker)


def load_model(model_path: str, cfg_path: str, device: str = "cpu") -> HookedTransformer:
    """
    Loads a saved model into HookedTransformer.

    Parameters
    ----------
    model_path : str
        Path to model's weights. (typically "model0.pt")
    cfg_path : str 
        Path to model's config. (typically "model_cfg.pt")
    device : str
        Device to load the model on
    """
    if not os.path.exists(model_path) and os.path.exists(cfg_path):
        raise ValueError("Path doesn't exist.")

    cfg = torch.load(cfg_path, weights_only=False, map_location=device)
    cfg.device = device

    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


if __name__ == "__main__":
    T0 = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0.5]
    ])
    T1 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0.5, 0, 0]
    ])

    dataset = MarkovData(n_gen=10000, gen_len=64, n_states=3, d_vocab=2, T_list=[T0, T1])

    if os.path.exists("./toy_transformer_checkpoints/model0.pt"):
        model = load_model("./toy_transformer_checkpoints/model0.pt", 
                         "./toy_transformer_checkpoints/model_cfg.pt")
    else:
        model = train_model(
            dataset=dataset, 
            n_epochs=5, 
            save_every=1000, 
            print_every=1000, 
            save_dir="./toy_transformer_checkpoints",
            # Enable ALL advanced metrics
            track_ngrams=True,
            ngram_orders=[2, 3],
            track_composition=True,
            track_previous_token=True,
            track_in_context=True,
            track_prefix_matching=True,
            wandb=True,
            wandb_project_name="advanced_toy_transformer"
        )

    model2 = finetune_model(
        model, dataset, 5, 
        save_dir=None,
        # Enable ALL advanced metrics for finetuning too
        track_ngrams=True,
        track_composition=True,
        track_previous_token=True,
        track_in_context=True,
        track_prefix_matching=True
    )

    # Test inference
    logits = model(torch.tensor([[0,1,1,0,1,0,0,1,1,0], 
                                [1,0,1,1,0,1,0,0,1,1], 
                                [1,0,0,1,0,0,1,0,0,1]], dtype=torch.int64))
    print(logits[:, -1])
    print(logits[:, -1].argmax(dim=-1))

    # Sample and compare
    sample, states = dataset.model.sample_sequence(max_new_tokens=40)
    preds = model(torch.tensor([sample], dtype=torch.int64)).argmax(dim=-1).flatten().tolist()

    for s, pred in zip(sample[1:], preds[:-1]):
        print(f"Actual: {s}, Predicted: {pred}")
