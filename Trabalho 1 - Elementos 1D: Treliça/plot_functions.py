import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from FEM_classes_3D import FEMSystem


import matplotlib.cm as cm
import matplotlib.colors as mcolors


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
    use_hue: bool = False,  # <-- Novo gatilho para o gradiente de cores
    plot_original: bool = True,
):
    """
    Plota a forma modal da treliça para um modo específico, com suporte a cores mapeadas.
    """
    mode_shape = eigenvectors[:, mode_index]

    max_idx = np.argmax(np.abs(mode_shape))
    if mode_shape[max_idx] < 0:
        mode_shape = -mode_shape

    u_full = mode_shape

    # ==========================================
    # CONFIGURAÇÃO DO MAPA DE CORES (HUE) EM Y
    # ==========================================
    if use_hue:
        # Pega todos os graus de liberdade em Y da estrutura
        y_dofs = [no.dof_y for no in system.nodes]
        y_disps = u_full[y_dofs]

        # Encontra o maior deslocamento (em módulo) para centralizar a escala no zero
        max_abs_y = np.max(np.abs(y_disps))
        # Se for muito próximo de zero (ex: modo apenas de deslocamento em X), evita divisão por zero
        if max_abs_y < 1e-12:
            max_abs_y = 1e-12

        # Normalizador de limites: vai de -max até +max
        norm = mcolors.Normalize(vmin=-max_abs_y, vmax=max_abs_y)
        # O colormap 'coolwarm' é perfeito: Vermelho (Y+), Branco (Y=0), Azul (Y-)
        cmap = cm.gnuplot

        # Gera a barra de cores na lateral da figura
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = axs.figure.colorbar(sm, ax=axs, fraction=0.046, pad=0.04)
        cbar.set_label("Deslocamento Relativo em Y", rotation=270, labelpad=15)

    # ==========================================
    # PLOTAGEM DOS ELEMENTOS
    # ==========================================
    for i, el in enumerate(system.elements):
        ix, iy = el.node_i.x, el.node_i.y
        jx, jy = el.node_j.x, el.node_j.y

        uix, uiy = u_full[el.node_i.dof_x], u_full[el.node_i.dof_y]
        ujx, ujy = u_full[el.node_j.dof_x], u_full[el.node_j.dof_y]

        ix_def, iy_def = ix + scale * uix, iy + scale * uiy
        jx_def, jy_def = jx + scale * ujx, jy + scale * ujy

        # Define a cor da linha: gradiente ou cor sólida fixa
        if use_hue:
            # A cor da barra será baseada na média do deslocamento Y dos seus dois nós
            avg_y = (uiy + ujy) / 2.0
            c_line = cmap(norm(avg_y))
        else:
            c_line = color

        # Plota estrutura original (apenas na primeira rodada para não repetir legenda)
        if plot_original:
            if i == 0:
                _ = axs.plot(
                    [ix, jx], [iy, jy], "k--", alpha=0.2, label="Não Deformada"
                )
            else:
                _ = axs.plot([ix, jx], [iy, jy], "k--", alpha=0.2)

        # Plota estrutura deformada
        lbl = label if i == 0 else ""
        _ = axs.plot(
            [ix_def, jx_def],
            [iy_def, jy_def],
            color=c_line,
            linestyle=linestyle,
            marker="o",
            linewidth=2,
            label=lbl,
            zorder=3
            if use_hue
            else 2,  # Garante que a linha colorida fique por cima
        )

    _ = axs.set_title(
        f"Modo {mode_index + 1}: f = {frequencies[mode_index]:.4f} Hz"
    )
    _ = axs.set_xlabel("X")
    _ = axs.set_ylabel("Y")
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
    Cria uma animação da forma modal da treliça.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    mode_shape = eigenvectors[:, mode_index]

    # Reconstrói o vetor de deslocamentos completo
    u_full = np.zeros(system.total_dofs)
    if system.boundarycond is not None:
        all_dofs = np.arange(system.total_dofs)
        free_dofs = np.delete(all_dofs, system.boundarycond)
        u_full[free_dofs] = mode_shape
    else:
        u_full = mode_shape

    # 1. Plota a estrutura original estática (fundo tracejado)
    for el in system.elements:
        ix, iy = el.node_i.x, el.node_i.y
        jx, jy = el.node_j.x, el.node_j.y
        _ = ax.plot([ix, jx], [iy, jy], "k--", alpha=0.2)

    # 2. Prepara os objetos de linha da estrutura deformada (iniciam vazios)
    lines = []
    for _ in system.elements:
        (line,) = ax.plot([], [], "b-o", linewidth=2)
        lines.append(line)

    ax.set_title(f"Modo {mode_index + 1}: f = {frequencies[mode_index]:.4f} Hz")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.grid(True)

    # 3. Função que atualiza o gráfico em cada frame
    def update(frame_angle):
        # O fator de tempo simula a oscilação harmônica (vai de 0, passa por 1, 0, -1 e volta a 0)
        time_factor = np.sin(frame_angle)

        for idx, el in enumerate(system.elements):
            ix, iy = el.node_i.x, el.node_i.y
            jx, jy = el.node_j.x, el.node_j.y

            uix, uiy = u_full[el.node_i.dof_x], u_full[el.node_i.dof_y]
            ujx, ujy = u_full[el.node_j.dof_x], u_full[el.node_j.dof_y]

            # Multiplica a escala e os deslocamentos pelo fator de tempo atual
            ix_def = ix + scale * uix * time_factor
            iy_def = iy + scale * uiy * time_factor
            jx_def = jx + scale * ujx * time_factor
            jy_def = jy + scale * ujy * time_factor

            # Atualiza os dados da linha específica
            lines[idx].set_data([ix_def, jx_def], [iy_def, jy_def])

        return lines

    # 4. Inicia o motor de animação passando de 0 a 2*pi
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=np.linspace(0, 2 * np.pi, frames),
        interval=50,
        blit=True,
    )

    plt.show()
    return ani
