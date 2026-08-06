# Genera el diagrama de arquitectura de la solución EcoSort AI (Actividad: Diseño de la solución de IA)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(11, 8.3))
ax.set_xlim(0, 11)
ax.set_ylim(0, 10.4)
ax.axis("off")

HW = "#DCE6F1"
HW_EDGE = "#4472A8"
SW = "#E2EFDA"
SW_EDGE = "#548235"
DEC = "#FCE4D6"
DEC_EDGE = "#C55A11"

def box(x, y, w, h, text, face, edge, fontsize=10.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            zorder=3, wrap=True)
    return (x, y, w, h)

def arrow(b1, b2, label=None, side="h"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    if side == "h":
        p1 = (x1 + w1, y1 + h1 / 2)
        p2 = (x2, y2 + h2 / 2)
    else:
        p1 = (x1 + w1 / 2, y1)
        p2 = (x2 + w2 / 2, y2 + h2)
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.6, color="#333333", zorder=1)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.18, label, ha="center", va="bottom", fontsize=8.5, color="#555555")

# Fila principal (pipeline izquierda -> derecha)
b_camara   = box(0.3, 6.6, 1.7, 1.3, "Cámara USB\n(hardware)", HW, HW_EDGE)
b_captura  = box(2.35, 6.6, 1.7, 1.3, "Captura de\nimagen\n(Raspberry Pi 5)", HW, HW_EDGE)
b_preproc  = box(4.4, 6.6, 1.7, 1.3, "Preprocesamiento\nresize 224x224\nnormalización", SW, SW_EDGE)
b_modelo   = box(6.45, 6.6, 1.9, 1.3, "Modelo IA\nMobileNetV2\n(TensorFlow Lite)", SW, SW_EDGE)
b_clasif   = box(8.7, 6.6, 2.0, 1.3, "Clasificación\nP(plástico) / P(vidrio)\n+ confianza", SW, SW_EDGE)

arrow(b_camara, b_captura)
arrow(b_captura, b_preproc)
arrow(b_preproc, b_modelo)
arrow(b_modelo, b_clasif)

# Bajada hacia el sistema experto
b_experto = box(7.6, 4.3, 3.0, 1.3, "Sistema Experto\nreglas de confianza\n(>=85% / 60-84% / <60%)", DEC, DEC_EDGE)
arrow(b_clasif, b_experto, side="v")

# Decisión final y ramas
b_decision = box(7.9, 2.2, 2.4, 1.1, "Decisión final\n(clase validada)", DEC, DEC_EDGE)
arrow(b_experto, b_decision, side="v")

b_dashboard = box(4.6, 0.4, 2.6, 1.0, "Dashboard web\n(registro / visualización)", HW, HW_EDGE)
b_recaptura = box(0.3, 2.2, 3.0, 1.1, "Solicitar recaptura\n(confianza 60-84%)", DEC, DEC_EDGE)
b_desconocido = box(0.3, 0.4, 3.0, 1.1, "Clase 'desconocido'\n(confianza <60%)", DEC, DEC_EDGE)

# Flechas desde sistema experto/decisión hacia las 3 salidas
a1 = FancyArrowPatch((7.9, 2.75), (3.3, 2.75), arrowstyle="-|>", mutation_scale=16,
                      linewidth=1.4, color="#888888", linestyle="--", zorder=1,
                      connectionstyle="arc3,rad=0.15")
ax.add_patch(a1)
ax.text(5.5, 3.05, "confianza media", ha="center", fontsize=8, color="#666666")

a2 = FancyArrowPatch((7.9, 2.4), (3.3, 0.95), arrowstyle="-|>", mutation_scale=16,
                      linewidth=1.4, color="#888888", linestyle="--", zorder=1,
                      connectionstyle="arc3,rad=0.2")
ax.add_patch(a2)
ax.text(5.5, 1.3, "confianza baja", ha="center", fontsize=8, color="#666666")

arrow(b_decision, b_dashboard, side="v")

ax.text(5.5, 10.05, "Arquitectura de la solución — EcoSort AI (clasificador plástico / vidrio)",
        ha="center", fontsize=13, fontweight="bold")

legend_elems = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=HW, markeredgecolor=HW_EDGE, markersize=14, label="Hardware / Integración física"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=SW, markeredgecolor=SW_EDGE, markersize=14, label="Procesamiento de IA (software)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=DEC, markeredgecolor=DEC_EDGE, markersize=14, label="Sistema experto / decisión"),
]
ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, 0.935),
          ncol=3, fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig("analysis/figs/04_arquitectura_sistema.png", dpi=200, bbox_inches="tight")
print("OK arquitectura")
