import wandb

# Inicializa un experimento
wandb.init(project="Prueba conexión wandb")

# Loggea datos (puede ser lo que sea)
for i in range(10):
    wandb.log({"mi_metrica": i})