import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def plot_2d_heatmap(
    system,
    scale: float = 1.0,
    title: str = "Mapa de Calor - Deslocamentos",
    cmap: str = "jet",
    figsize: tuple[int | float, int | float] = (10, 6),
):
    """
    Gera um mapa de calor (heatmap) preenchendo os elementos 2D com as cores
    correspondentes à magnitude do deslocamento.
    """
    # 1. Extração das coordenadas originais e deslocamentos dos nós
    n_nodes = len(system.nodes)
    x = np.zeros(n_nodes)
    y = np.zeros(n_nodes)
    ux = np.zeros(n_nodes)
    uy = np.zeros(n_nodes)

    # Criamos um dicionário reverso para achar a posição do nó no array rapidamente
    node_idx_map = {node.id: idx for idx, node in enumerate(system.nodes)}

    for node in system.nodes:
        idx = node_idx_map[node.id]
        x[idx] = node.x
        y[idx] = node.y
        ux[idx] = node.res_ux
        uy[idx] = node.res_uy

    # 2. Cálculo das coordenadas deformadas e da magnitude do deslocamento
    x_def = x + scale * ux
    y_def = y + scale * uy
    disp_magnitude = np.sqrt(ux**2 + uy**2)  # Deslocamento resultante

    # 3. Montagem da matriz de conectividade para o Matplotlib
    # O matplotlib exige uma lista de listas com os 3 índices dos vértices de cada triângulo
    triangles = []
    for el in system.elements:
        idx_1 = node_idx_map[el.nodes[0].id]
        idx_2 = node_idx_map[el.nodes[1].id]
        idx_3 = node_idx_map[el.nodes[2].id]
        triangles.append([idx_1, idx_2, idx_3])

    connectivity = np.array(triangles)

    # 4. Criação do objeto de Triangulação do Matplotlib
    triangulation_orig = mtri.Triangulation(x, y, connectivity)
    triangulation_def = mtri.Triangulation(x_def, y_def, connectivity)

    # 5. Plotagem
    fig, ax = plt.subplots(figsize=figsize)

    # Plota a malha original como fundo (linhas cinzas tracejadas)
    ax.triplot(
        triangulation_orig, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
    )

    # Plota o preenchimento de cor (heatmap) na malha deformada
    # tricontourf interpola os valores nodais e cria zonas de cor suaves
    contour = ax.tricontourf(
        triangulation_def, disp_magnitude, levels=20, cmap=cmap, alpha=0.9
    )

    # Opcional: Desenha as arestas pretas por cima do mapa de calor para destacar os elementos
    ax.triplot(triangulation_def, color="black", linewidth=0.5, alpha=0.8)

    # Barra de cores lateral
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Magnitude do Deslocamento [m]", rotation=270, labelpad=15)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    plt.grid(True, linestyle=":", alpha=0.4)

    return fig, ax
