from tkinter import filedialog as fd
import pickle
import matplotlib.pyplot as plt

pickle_fig = fd.askopenfilename(title="Please select ABF figure:",
                                filetypes=(("PICKLE files", "*.pickle"), ("all files", "*.*")))
print(f"Figure file: {pickle_fig}")

with open(pickle_fig, 'rb') as f:
    fig_abf = pickle.load(f)

plt.show()

