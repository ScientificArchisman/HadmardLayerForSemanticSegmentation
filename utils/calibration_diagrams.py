import torch, numpy as np
import matplotlib.pyplot as plt

def plot_reliability_diagram(ece_out, title_prefix="", output_path=None):
    bc = ece_out["bin_counts"].detach().cpu().numpy()
    ba = ece_out["bin_acc"].detach().cpu().numpy()
    bf = ece_out["bin_conf"].detach().cpu().numpy()
    ed = ece_out["edges"].detach().cpu().numpy()
    ece = float(ece_out.get("ece", torch.tensor(0.)).detach().cpu().item())

    # Bin centers and widths
    centers = 0.5 * (ed[:-1] + ed[1:])
    widths = (ed[1:] - ed[:-1])

    # Mask empty bins to avoid clutter
    m = bc > 0
    centers, widths, ba, bf, bc = centers[m], widths[m], ba[m], bf[m], bc[m]

    plt.figure()
    plt.bar(centers, ba, width=widths, align="center", alpha=0.8, edgecolor="black")
    xs = np.linspace(0, 1, 101) # perfect calibration line
    plt.plot(xs, xs)
    plt.plot(centers, bf, marker="o", linestyle="")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence (bin)")
    plt.ylabel("Empirical accuracy")
    plt.title(f"{title_prefix}Reliability diagram  (ECE={ece:.4f})")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)




def plot_gap_bars(ece_out, title_prefix="", output_path=None):
    bc = ece_out["bin_counts"].detach().cpu().numpy()
    ba = ece_out["bin_acc"].detach().cpu().numpy()
    bf = ece_out["bin_conf"].detach().cpu().numpy()
    ed = ece_out["edges"].detach().cpu().numpy()

    centers = 0.5 * (ed[:-1] + ed[1:])
    widths = (ed[1:] - ed[:-1])

    m = bc > 0
    centers, widths, ba, bf, bc = centers[m], widths[m], ba[m], bf[m], bc[m]
    gap = np.abs(ba - bf)

    plt.figure()
    plt.bar(centers, gap, width=widths, align="center", edgecolor="black")
    plt.xlabel("Confidence (bin)")
    plt.ylabel("|Accuracy − Confidence|")
    plt.title(f"{title_prefix}Per-bin calibration gap")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)



def plot_confidence_histogram(ece_out, title_prefix="", output_path=None):
    bc = ece_out["bin_counts"].detach().cpu().numpy()
    ed = ece_out["edges"].detach().cpu().numpy()

    widths = (ed[1:] - ed[:-1])
    mass = bc.astype(float) / max(bc.sum(), 1)  # fraction of pixels per bin

    plt.figure()
    plt.bar(0.5*(ed[:-1]+ed[1:]), mass, width=widths, align="center", edgecolor="black")
    plt.xlabel("Confidence (bin)")
    plt.ylabel("Fraction of pixels")
    plt.title(f"{title_prefix}Confidence distribution")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)




def plot_acc_conf_lines(ece_out, title_prefix="", output_path=None):
    bc = ece_out["bin_counts"].detach().cpu().numpy()
    ba = ece_out["bin_acc"].detach().cpu().numpy()
    bf = ece_out["bin_conf"].detach().cpu().numpy()
    ed = ece_out["edges"].detach().cpu().numpy()

    centers = 0.5 * (ed[:-1] + ed[1:])
    m = bc > 0
    centers, ba, bf = centers[m], ba[m], bf[m]

    xs = np.linspace(0, 1, 101)

    plt.figure()
    plt.plot(xs, xs)                 # perfect calibration
    plt.plot(centers, ba, marker="o")
    plt.plot(centers, bf, marker="s")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence (bin center)")
    plt.ylabel("Value")
    plt.title(f"{title_prefix}Accuracy and confidence by bin")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
