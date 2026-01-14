import matplotlib.pyplot as plt
from torchinfo import summary
from rich.console import Console
console = Console()

def visualize(model, model_name, input_data):
    out = model(input_data)
    console.print(f'Computed output, shape = {out.shape=}')
    model_stats = summary(model,
                          input_data=input_data,
                          col_names=[
                              "input_size",
                              "output_size",
                              "num_params",
                              # "params_percent",
                              # "kernel_size",
                              # "mult_adds",
                          ],
                          row_settings=("var_names",),
                          col_width=18,
                          depth=8,
                          verbose=0,
                          )
    console.print(model_stats)