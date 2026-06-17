import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from FEM_classes_3D import FEMSystem


def plot_mode_shape(
    system: FEMSystem,
    frequencies: npt.NDArray[np.float64],
    eigenvectors: npt.NDArray[np.float64],
    axs: plt.Axes,
    mode_index: int = 0,
    scale: float = 0.1,
    label: str = "Custom Solver",
    color: str = "blue",
    linestyle: str = "-",
    use_hue: bool = False,
    plot_original: bool = True,
):
    """
    Plota a forma modal usando os resultados armazenados diretamente nos nós.
    """
    # 1. Extrai o vetor do modo e inverte o sinal se necessário (padronização visual)
    mode_shape = eigenvectors[:, mode_index]
    max_idx = np.argmax(np.abs(mode_shape))
    if mode_shape[max_idx] < 0:
        mode_shape = -mode_shape

    # 2. A MÁGICA ACONTECE AQUI: Alimenta os nós com os resultados deste modo
    system.map_results_to_nodes(mode_shape)

    cmap = None
    norm = None

    # 3. Configuração do mapa de cores (baseado na magnitude do deslocamento máximo)
    if use_hue:
        # Extrai a magnitude do deslocamento total (3D) de cada nó
        disps = [
            np.sqrt(no.res_ux**2 + no.res_uy**2 + no.res_uz**2) for no in system.nodes
        ]

        max_disp = np.max(disps)
        if max_disp < 1e-12:
            max_disp = 1e-12  # Evita divisão por zero

        norm = mcolors.Normalize(vmin=0, vmax=max_disp)
        cmap = plt.get_cmap("gnuplot")

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = axs.figure.colorbar(sm, ax=axs, fraction=0.046, pad=0.04)
        cbar.set_label("Magnitude do Deslocamento Relativo", rotation=270, labelpad=15)

    # 4. Plotagem dos Elementos
    for i, el in enumerate(system.elements):
        ix, iy = el.node_i.x, el.node_i.y
        jx, jy = el.node_j.x, el.node_j.y

        # Leitura limpa diretamente do objeto Nó, sem fatiar vetores com índices!
        uix, uiy, uiz = el.node_i.res_ux, el.node_i.res_uy, el.node_i.res_uz
        ujx, ujy, ujz = el.node_j.res_ux, el.node_j.res_uy, el.node_j.res_uz

        ix_def, iy_def = ix + scale * uix, iy + scale * uiy
        jx_def, jy_def = jx + scale * ujx, jy + scale * ujy

        if use_hue and cmap is not None and norm is not None:
            disp_i = np.sqrt(uix**2 + uiy**2 + uiz**2)
            disp_j = np.sqrt(ujx**2 + ujy**2 + ujz**2)
            avg_disp = (disp_i + disp_j) / 2.0
            c_line = cmap(norm(avg_disp))
        else:
            c_line = color

        if plot_original:
            if i == 0:
                axs.plot([ix, jx], [iy, jy], "k--", alpha=0.2, label="Não Deformada")
            else:
                axs.plot([ix, jx], [iy, jy], "k--", alpha=0.2)

        lbl = label if i == 0 else ""
        axs.plot(
            [ix_def, jx_def],
            [iy_def, jy_def],
            color=c_line,
            linestyle=linestyle,
            marker="o",
            linewidth=2,
            label=lbl,
            zorder=3 if use_hue else 2,
        )

    axs.set_title(f"Modo {mode_index + 1}: f = {frequencies[mode_index]:.4f} Hz")
    axs.set_xlabel("X [m]")
    axs.set_ylabel("Y [m]")
    axs.grid(True, linestyle=":", alpha=0.6)
    axs.legend(loc="upper right")


def animate_mode_shape(
    system: FEMSystem,
    frequencies: npt.NDArray[np.float64],
    eigenvectors: npt.NDArray[np.float64],
    mode_index: int = 0,
    scale: float = 0.1,
    frames: int = 60,
):
    """
    Cria uma animação da forma modal lendo os deslocamentos diretamente dos nós.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepara os nós com os resultados do modo
    mode_shape = eigenvectors[:, mode_index]
    system.map_results_to_nodes(mode_shape)

    # 1. Plota a estrutura original estática
    for el in system.elements:
        ax.plot(
            [el.node_i.x, el.node_j.x], [el.node_i.y, el.node_j.y], "k--", alpha=0.2
        )

    # 2. Prepara as linhas para a animação
    lines = []
    for _ in system.elements:
        (line,) = ax.plot([], [], "b-o", linewidth=2)
        lines.append(line)

    ax.set_title(f"Modo {mode_index + 1}: f = {frequencies[mode_index]:.4f} Hz")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.grid(True)

    # 3. Função de atualização super otimizada
    def update(frame_angle):
        time_factor = np.sin(frame_angle)

        for idx, el in enumerate(system.elements):
            ix, iy = el.node_i.x, el.node_i.y
            jx, jy = el.node_j.x, el.node_j.y

            # Lemos diretamente do nó!
            uix, uiy = el.node_i.res_ux, el.node_i.res_uy
            ujx, ujy = el.node_j.res_ux, el.node_j.res_uy

            ix_def = ix + scale * uix * time_factor
            iy_def = iy + scale * uiy * time_factor
            jx_def = jx + scale * ujx * time_factor
            jy_def = jy + scale * ujy * time_factor

            lines[idx].set_data([ix_def, jx_def], [iy_def, jy_def])

        return lines

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=np.linspace(0, 2 * np.pi, frames),
        interval=50,
        blit=True,
    )
    plt.show()
    return ani
