from tqdm import tqdm

def create_progress_bar(dataloader, desc='Training'):
    return tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=False)

def update_progress_bar(bar, loss):
    bar.set_postfix({
        "loss": f"{loss:.4f}"
    })
    bar.update(1)
