import torch

device = torch.device("cuda")

x = torch.rand(4000, 4000, device=device)
y = torch.rand(4000, 4000, device=device)

z = x @ y

print("GPU :", torch.cuda.get_device_name(0))
print("Calcul effectué sur :", z.device)
print("Résultat :", z.mean().item())