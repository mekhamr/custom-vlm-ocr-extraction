import wandb

wandb.login(key='4ee4cd731d923c63f088f74980bc494af578c66b')  
wandb.init(project="custom-vlm-ocr", name="florence-2-finetune")
print("WandB initialized.")
