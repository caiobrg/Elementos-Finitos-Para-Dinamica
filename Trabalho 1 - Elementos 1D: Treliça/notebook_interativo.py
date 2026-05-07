import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dinâmica Estrutural e Análise Modal de Treliças (MEF)

    A análise dinâmica de uma estrutura começa pela compreensão de suas propriedades intrínsecas: os modos de vibrar e suas frequências naturais. O Método dos Elementos Finitos (MEF) discretiza o domínio contínuo da treliça em um número finito de graus de liberdade, transformando as equações diferenciais parciais em um sistema algébrico matricial.

    A equação de movimento para o sistema em vibração livre e não amortecida é governada pelo equilíbrio entre as forças inerciais e as forças elásticas:

    $$[M]\{\ddot{U}\} + [K]\{U\} = \{0\}$$

    Onde $[M]$ é a matriz de massa global e $[K]$ é a matriz de rigidez global. Assumindo que a estrutura vibra de forma harmônica simples sincronizada, os deslocamentos podem ser descritos como $\{U\} = \{\phi\}e^{i\omega t}$. Substituindo essa solução na equação de movimento, chegamos ao problema de autovalores:

    $$([K] - \omega^2[M])\{\phi\} = \{0\}$$

    A resolução deste sistema nos fornece duas informações físicas fundamentais:
    1. **Autovalores ($\omega^2$):** Representam as frequências circulares naturais ao quadrado. A partir delas, obtemos as frequências naturais em Hertz ($f = \omega / 2\pi$), que indicam as frequências de ressonância da estrutura.
    2. **Autovetores ($\{\phi\}$):** Representam as Formas Modais, ou seja, a configuração geométrica relativa que a estrutura assume ao vibrar em cada uma dessas frequências naturais.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Importamos todas as classes criadas, a configuração da treliça e nosso helper script para plotagem, o OpenSees e algumas outras bibliotecas necessárias para a análise.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy.typing as npt
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import eigsh, spsolve
    import pandas as pd

    return cm, eigsh, lil_matrix, mcolors, mo, np, npt, pd, plt, spsolve


@app.cell
def _(eigsh, lil_matrix, np, npt, spsolve):
    """
    Implementação 3D (3 GDL por nó) de elementos de treliça para MEF
    """


    class Node:
        """
        Representa um nó em uma estrutura de treliça 2D.

        Atributos:
            id (int): Identificador único do nó.
            x (float): Coordenada X do nó.
            y (float): Coordenada Y do nó.
            z (float): Coordenada Z do nó.
            dof_x (int): Índice global do grau de liberdade na direção X.
            dof_y (int): Índice global do grau de liberdade na direção Y.
            dof_Z (int): Índice global do grau de liberdade na direção Z.
        """

        def __init__(self, id: int, x: float, y: float, z: float):
            self.id: int = id
            self.x: float = x
            self.y: float = y
            self.z: float = z

            self.dof_x: int = 3 * id
            self.dof_y: int = 3 * id + 1
            self.dof_z: int = 3 * id + 2


    class TrussElement:
        """
        Representa um elemento de barra em uma estrutura de treliça 2D.

        Atributos:
            id (int): Identificador único do elemento.
            node_i (Node): Nó inicial do elemento.
            node_j (Node): Nó final do elemento.
            E (float): Módulo de elasticidade do material.
            A (float): Área da seção transversal do elemento.
            rho (float): Densidade do material.
            L (float): Comprimento do elemento.
            c (float): Cosseno do ângulo com o eixo X.
            s (float): Seno do ângulo com o eixo X.
            dofs (list): Lista dos índices globais de GDL associados a este elemento.
        """

        def __init__(
            self,
            id: int,
            node_i: Node,
            node_j: Node,
            E: float,
            A: float,
            rho: float,
        ) -> None:
            self.id: int = id
            self.node_i: Node = node_i
            self.node_j: Node = node_j
            self.E: float = E
            self.A: float = A
            self.rho: float = rho

            # Deltas em 3D
            dx: float = node_j.x - node_i.x
            dy: float = node_j.y - node_i.y
            dz: float = node_j.z - node_i.z

            self.L: float = np.sqrt(dx**2 + dy**2 + dz**2)

            # Cossenos diretores (Vetor de orientação)
            self.v: npt.NDArray[np.float64] = np.array(
                [dx / self.L, dy / self.L, dz / self.L]
            )

            # Vetor contendo os 6 índices globais de GDL para este elemento
            self.dofs: list[int] = [
                node_i.dof_x,
                node_i.dof_y,
                node_i.dof_z,
                node_j.dof_x,
                node_j.dof_y,
                node_j.dof_z,
            ]

        def get_stiffness_matrix(self) -> npt.NDArray[np.float64]:
            coef: float = (self.E * self.A) / self.L

            # Produto externo do vetor v cria a submatriz 3x3 dos cossenos
            # Lambda = [[Cx^2,  CxCy,  CxCz],
            #           [CyCx,  Cy^2,  CyCz],
            #           [CzCx,  CzCy,  Cz^2]]
            Lambda: npt.NDArray[np.float64] = np.outer(self.v, self.v)

            # Constrói a matriz 6x6 usando blocos (limpo e sem erro de digitação)
            K_e_bar: npt.NDArray[np.float64] = coef * np.block(
                [[Lambda, -Lambda], [-Lambda, Lambda]]
            )
            return K_e_bar

        def get_mass_matrix(self) -> npt.NDArray[np.float64]:
            coef: float = (self.rho * self.A * self.L) / 6
            I3 = np.eye(3)  # Matriz Identidade 3x3

            # A matriz de massa consistente em 3D é distribuída simetricamente
            M_e_bar = coef * np.block([[2 * I3, I3], [I3, 2 * I3]])
            return M_e_bar


    class FEMSystem:
        """
        Gerencia a montagem e análise do sistema via Método dos Elementos Finitos (MEF).

        Atributos:
            nodes (list[Node]): Lista de nós no sistema.
            elements (list[TrussElement]): Lista de elementos no sistema.
            boundarycond (list[int]): Índices dos GDLs com restrição.
            total_dofs (int): Número total de graus de liberdade no sistema.
        """

        def __init__(
            self,
            nodes: list[Node],
            elements: list[TrussElement],
            boundarycond: list[int] | None = None,
        ) -> None:
            self.nodes: list[Node] = nodes
            self.elements: list[TrussElement] = elements
            self.boundarycond: list[int] = (
                boundarycond if boundarycond is not None else []
            )
            self.total_dofs: int = len(nodes) * 3

            self._update_free_dofs()

            # Dicionário para armazenar as forças {grau_de_liberdade: magnitude}
            self.nodal_loads: dict[int, float] = {}

        def _update_free_dofs(self) -> None:
            """Atualiza o array de GDLs livres baseado nas condições de contorno."""
            all_dofs: npt.NDArray[np.int32] = np.arange(
                self.total_dofs, dtype="int32"
            )
            if self.boundarycond:
                # Remove duplicatas e ordena para garantir consistência
                unique_bc = sorted(list(set(self.boundarycond)))
                self.free_dofs_array: npt.NDArray[np.int32] = np.delete(
                    all_dofs, unique_bc
                )
            else:
                self.free_dofs_array = all_dofs

        def apply_boundary_condition(
            self, node_id: int, direction: str | list[str]
        ) -> None:
            """
            Restringe o deslocamento de um nó em uma ou mais direções.
            direction: 'X', 'Y', 'Z' ou uma lista como ['X', 'Y']
            """
            no = next((n for n in self.nodes if n.id == node_id), None)
            if no is None:
                raise ValueError(f"Nó {node_id} não encontrado.")

            if isinstance(direction, str):
                directions = [direction.upper()]
            else:
                directions = [d.upper() for d in direction]

            for d in directions:
                if d == "X":
                    dof = no.dof_x
                elif d == "Y":
                    dof = no.dof_y
                elif d == "Z":
                    dof = no.dof_z
                else:
                    raise ValueError(f"Direção inválida: {d}. Deve ser X, Y ou Z.")

                if dof not in self.boundarycond:
                    self.boundarycond.append(dof)

            self._update_free_dofs()

        def apply_load(self, node_id: int, direction: str, magnitude: float):
            """
            Aplica uma força harmônica em um nó específico.
            direction: 'X', 'Y' ou 'Z'
            """

            # Percorre a lista de nós para encontrar um ID que satisfaça o node_id
            no = next((n for n in self.nodes if n.id == node_id), None)
            if no is None:
                raise ValueError(f"Nó {node_id} não encontrado.")

            if direction.upper() == "X":
                dof = no.dof_x
            elif direction.upper() == "Y":
                dof = no.dof_y
            elif direction.upper() == "Z":
                dof = no.dof_z
            else:
                raise ValueError("Direção deve ser X, Y ou Z.")

            self.nodal_loads[dof] = self.nodal_loads.get(dof, 0.0) + magnitude

        def assemble_stiffness(self):
            """
            Monta a matriz de rigidez global e aplica as condições de contorno.

            Retorna:
                scipy.sparse.csr_matrix: Matriz de rigidez global montada.
            """
            # Inicializa a matriz global usando o formato LIL para construção eficiente
            K_global: lil_matrix = lil_matrix((self.total_dofs, self.total_dofs))

            for el in self.elements:
                K_e: npt.NDArray[np.float64] = el.get_stiffness_matrix()
                dofs: list[int] = el.dofs

                # Utiliza np.ix_ para criar a grade de índices para adicionar a matriz 4x4 do elemento
                grid: tuple[npt.NDArray[np.intp], ...] = np.ix_(dofs, dofs)
                K_global[grid] += K_e  # pyright: ignore[reportCallIssue, reportArgumentType]

            # Aplica condições de contorno removendo os GDLs restritos
            K_global = K_global[np.ix_(self.free_dofs_array, self.free_dofs_array)]  # pyright: ignore[reportCallIssue, reportArgumentType]

            # Converte para o formato CSR para operações numéricas eficientes
            return K_global.tocsr()

        def assemble_mass(self):
            """
            Monta a matriz de massa global e aplica as condições de contorno.

            Retorna:
                scipy.sparse.csr_matrix: Matriz de massa global montada.
            """
            M_global: lil_matrix[np.float64] = lil_matrix(
                (self.total_dofs, self.total_dofs)
            )

            for el in self.elements:
                M_e: npt.NDArray[np.float64] = el.get_mass_matrix()
                grid = np.ix_(el.dofs, el.dofs)
                M_global[grid] += M_e  # pyright: ignore[reportCallIssue, reportArgumentType]

            # Aplica condições de contorno
            M_global = M_global[np.ix_(self.free_dofs_array, self.free_dofs_array)]  # pyright: ignore[reportCallIssue, reportArgumentType]

            return M_global.tocsr()

        def solve_modal_analysis(
            self, num_modes=6
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            """
            Calcula as frequências naturais e as formas modais do sistema.

            Argumentos:
                num_modes (int): Número de modos de vibração a calcular.

            Retorna:
                tuple: (frequencias_naturais, formas_modais)
                    - frequencias_naturais (npt.NDArray): Frequências em Hz.
                    - formas_modais (npt.NDArray): Matriz com os autovetores correspondentes.
            """
            K = self.assemble_stiffness()
            M = self.assemble_mass()

            # 1. Identificar os graus de liberdade livres corretamente como um array de índices
            num_modes = min(num_modes, len(self.free_dofs_array) - 1)

            # 2. Resolve o problema de autovalor usando matrizes ESPARSAS
            eigenvalues_free, eigenvectors_free = eigsh(
                K, k=num_modes, M=M, sigma=0.1
            )

            # Fatiar para pegar apenas a quantidade de modos solicitados
            eigenvalues_free = eigenvalues_free[:num_modes]
            eigenvectors_free = eigenvectors_free[:, :num_modes]

            # 3. REMONTAGEM: De 19 para 33 linhas
            # Cria a matriz cheia de zeros com o tamanho TOTAL do sistema 3D
            eigenvectors = np.zeros((self.total_dofs, num_modes))

            # Atribuição direta limpa (agora free_dofs_array tem exatamente 19 índices válidos)
            eigenvectors[self.free_dofs_array, :] = eigenvectors_free

            # 4. Ordenação e conversão para Hz
            idx = eigenvalues_free.argsort()
            eigenvalues = eigenvalues_free[idx]
            eigenvectors = eigenvectors[:, idx]

            natural_frequencies = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)

            return natural_frequencies, eigenvectors

        def solve_harmonic_analysis(
            self,
            freq_range_hz: npt.NDArray[np.float64],
            alpha_rayleigh: float = 0.0,
            beta_rayleigh: float = 0.0,
        ) -> npt.NDArray[np.complex128]:
            """
            Resolve a resposta em frequência (harmônica) do sistema.
            Retorna uma matriz complexa (GDLs x Frequências).
            """

            K = self.assemble_stiffness()
            M = self.assemble_mass()

            # Matriz de amortecimento de Rayleigh
            C = alpha_rayleigh * M + beta_rayleigh * K

            # Monta o vetor de forças global
            F_global = np.zeros(self.total_dofs, dtype=np.complex128)
            for dof, mag in self.nodal_loads.items():
                F_global[dof] = mag

            # Corta o vetor de forças para os GDLs livres, complex128 para acelerar o spsolve e permitir implementação de fase no forçamento
            F_free = F_global[self.free_dofs_array]

            # Matriz para guardar todos os resultados (N° GDL linhas x N frequências)
            # complex128 pois a resposta tem amplitude e fase
            respostas = np.zeros(
                (self.total_dofs, len(freq_range_hz)), dtype=np.complex128
            )

            # Varredura de frequências
            for i, f in enumerate(freq_range_hz):
                omega = 2.0 * np.pi * f

                # Matriz de Impedância Dinâmica: [H] = [K] - w^2[M] + iw[C]
                H = K - (omega**2) * M + (1j * omega) * C

                # Resolve o sistema linear esparso para esta frequência
                u_free = spsolve(H, F_free)

                # Remonta para a matriz global
                respostas[self.free_dofs_array, i] = u_free

            return respostas


    return FEMSystem, Node, TrussElement


@app.cell
def _(FEMSystem, cm, mcolors, np, npt, plt):
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

    return (plot_mode_shape,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuração da treliça
    """)
    return


@app.cell
def _():
    """
    Configuração da treliça para manter a padronização entre todos os métodos de solução
    """

    from dataclasses import dataclass


    @dataclass
    class material:
        modElast: float  # Modulo de Young
        coefPoiss: float
        dens: float  # Densidade
        A: float  # Área da seção
        name: str  # nome do material


    @dataclass
    class system:
        nodes: dict[
            int, tuple[float, float, float]
        ]  # dicionário de nós com id e tupla tupla de coordenadas (x,y,z)
        elem: dict[
            int, tuple[int, int]
        ]  # dicionário de elementos contendo id e uma tupla (nó_i, nó_j)
        material: material

    return material, system


@app.cell
def _(mo, pd):
    # 1. Dados iniciais (Template da sua treliça padrão)
    df_n_init = pd.DataFrame(
        [
            {"id": 0, "x": 0.0, "y": 0.0, "z": 0.0},
            {"id": 1, "x": 2.0, "y": 0.0, "z": 0.0},
            {"id": 2, "x": 4.0, "y": 0.0, "z": 0.0},
            {"id": 3, "x": 6.0, "y": 0.0, "z": 0.0},
            {"id": 4, "x": 8.0, "y": 0.0, "z": 0.0},
            {"id": 5, "x": 10.0, "y": 0.0, "z": 0.0},
            {"id": 6, "x": 1.0, "y": 2.0, "z": 0.0},
            {"id": 7, "x": 3.0, "y": 2.0, "z": 0.0},
            {"id": 8, "x": 5.0, "y": 2.0, "z": 0.0},
            {"id": 9, "x": 7.0, "y": 2.0, "z": 0.0},
            {"id": 10, "x": 9.0, "y": 2.0, "z": 0.0},
        ]
    )

    df_e_init = pd.DataFrame(
        [
            {"id": 0, "no_i": 0, "no_j": 1},
            {"id": 1, "no_i": 1, "no_j": 2},
            {"id": 2, "no_i": 2, "no_j": 3},
            {"id": 3, "no_i": 3, "no_j": 4},
            {"id": 4, "no_i": 4, "no_j": 5},
            {"id": 5, "no_i": 6, "no_j": 7},
            {"id": 6, "no_i": 7, "no_j": 8},
            {"id": 7, "no_i": 8, "no_j": 9},
            {"id": 8, "no_i": 9, "no_j": 10},
            {"id": 9, "no_i": 0, "no_j": 6},
            {"id": 10, "no_i": 6, "no_j": 1},
            {"id": 11, "no_i": 1, "no_j": 7},
            {"id": 12, "no_i": 7, "no_j": 2},
            {"id": 13, "no_i": 2, "no_j": 8},
            {"id": 14, "no_i": 8, "no_j": 3},
            {"id": 15, "no_i": 3, "no_j": 9},
            {"id": 16, "no_i": 9, "no_j": 4},
            {"id": 17, "no_i": 4, "no_j": 10},
            {"id": 18, "no_i": 10, "no_j": 5},
        ]
    )

    # 2. Criação dos Editores (Sintaxe simplificada para evitar AttributeError)
    # Qualquer alteração nestas tabelas disparará o recálculo do restante do notebook
    editor_n = mo.ui.data_editor(df_n_init, label="📍 Edite os Nós (X, Y, Z)")

    editor_n_com_titulo = mo.vstack([mo.md("Nós da Treliça"), editor_n])
    editor_e = mo.ui.data_editor(df_e_init, label="🔗 Edite as Barras (nó_i, nó_j)")
    editor_e_com_titulo = mo.vstack([mo.md("Elementos da Treliça"), editor_e])
    mo.hstack([editor_n_com_titulo, editor_e_com_titulo])
    return editor_e, editor_n


@app.cell
def _(editor_e, editor_n, material, system):
    nodes_dict = {
        int(r["id"]): (float(r["x"]), float(r["y"]), float(r["z"]))
        for _, r in editor_n.value.iterrows()
    }

    elem_dict = {
        int(r["id"]): (int(r["no_i"]), int(r["no_j"]))
        for _, r in editor_e.value.iterrows()
    }

    # 2. Recria o material (pode adicionar sliders aqui depois se quiser)
    aco_base = material(
        modElast=210e9, coefPoiss=0.3, dens=7850, A=0.005, name="ACO"
    )

    # 3. Gera o novo objeto de treliça dinâmico
    # A partir daqui, todas as outras células devem usar 'trelica_nova'
    trelica = system(nodes=nodes_dict, elem=elem_dict, material=aco_base)

    return (trelica,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analise modal

    Vamos primeiro montar o sistema no solver desenvolvido para o trabalho e fazer a análise modal do sistema.
    """)
    return


@app.cell
def _(FEMSystem, Node, TrussElement, trelica):
    # Quantidade de modos a serem calculados
    numModos = list(trelica.elem.keys())[-1]

    nodes: list[Node] = []
    for id, (x, y, z) in trelica.nodes.items():
        nodes.append(Node(id=id, x=x, y=y, z=z))

    elems: list[TrussElement] = []
    for id, (no_i, no_j) in trelica.elem.items():
        elems.append(
            TrussElement(
                id=id,
                node_i=nodes[no_i],
                node_j=nodes[no_j],
                E=trelica.material.modElast,
                A=trelica.material.A,
                rho=trelica.material.dens,
            )
        )

    bcs = [
        nodes[0].dof_x,  # Apoio fixo: Trava X
        nodes[0].dof_y,  # Apoio fixo: Trava Y
        nodes[5].dof_y,  # Apoio móvel: Trava Y
    ]

    # Força todos os nós a não saírem do plano 2D
    for no in nodes:
        bcs.append(no.dof_z)

    sistema = FEMSystem(nodes=nodes, elements=elems, boundarycond=bcs)

    freqNaturais, formasModais = sistema.solve_modal_analysis(num_modes=numModos)
    return formasModais, freqNaturais, numModos, sistema


@app.cell
def _(mo, numModos):
    seletor_escala = mo.ui.slider(
        start=0.01,
        stop=100,
        step=0.5,
        value=1,
        label="**Selecione a escala de visualização:**",
        show_value=True,
    )

    seletor_modo = mo.ui.slider(
        start=1,
        stop=numModos,
        step=1,
        value=1,
        label="**Selecione o Modo de Vibração:**",
        show_value=True,
    )


    controles = mo.vstack(
        [
            seletor_escala,
            seletor_modo,
        ]
    )
    return controles, seletor_escala, seletor_modo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualização das Formas Modais

    Fisicamente, as formas modais não possuem uma amplitude absoluta (elas não estão em milímetros ou metros). Elas representam apenas a proporção de deslocamento entre os nós da estrutura. Uma forma modal indica que, quando o nó A se desloca 1 unidade para cima, o nó B necessariamente se desloca 0.5 unidades para baixo, por exemplo.

    Para efeitos de visualização, aplicamos um fator de escala arbitrário. O objetivo desta representação é permitir a inspeção visual dos eixos de flexão e identificar quais nós permanecem parados (nós modais) durante a vibração em uma frequência específica.
    """)
    return


@app.cell
def _(
    controles,
    formasModais,
    freqNaturais,
    mo,
    plot_mode_shape,
    plt,
    seletor_escala,
    seletor_modo,
    sistema,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = seletor_modo.value - 1

    plot_mode_shape(
        sistema,
        freqNaturais,
        formasModais,
        axs=ax,
        mode_index=idx,
        scale=seletor_escala.value,
        label="Custom Solver",
        use_hue=True,
    )

    mo.vstack([controles, mo.as_html(fig)])
    return


@app.cell
def _(mo, sistema):
    # Range Slider para a Frequência (Min e Max)
    seletor_faixa_freq = mo.ui.range_slider(
        start=0.2,
        stop=1000.0,
        step=1.0,
        value=[0.2, 100.0],
        label="**Faixa de Frequência [Hz]:**",
    )

    # Nós com GDLs livres
    opcoes_nos = {}
    for n in sistema.nodes:
        if n.dof_x in sistema.free_dofs_array:
            opcoes_nos[f"Nó {n.id} (Direção X)"] = n.dof_x
        if n.dof_y in sistema.free_dofs_array:
            opcoes_nos[f"Nó {n.id} (Direção Y)"] = n.dof_y

    seletor_gdl = mo.ui.multiselect(
        options=opcoes_nos,
        value=[list(opcoes_nos.keys())[3]],
        label="**Selecione os Nós para Monitorar:**",
    )

    # Empilha os controles na tela
    controles_harm = mo.hstack([seletor_faixa_freq, seletor_gdl], justify="start")
    return controles_harm, opcoes_nos, seletor_faixa_freq, seletor_gdl


@app.cell
def _(mo, sistema):
    opcoes_nos_carga = {f"Nó {n.id}": n.id for n in sistema.nodes}

    # 1. Seletor Mestre de Quantidade
    seletor_qtd_cargas = mo.ui.number(
        start=1,
        stop=10,
        step=1,
        value=1,
        label="**Quantidade de Forças Simultâneas:**",
    )
    return opcoes_nos_carga, seletor_qtd_cargas


@app.cell
def _(mo, opcoes_nos_carga, seletor_qtd_cargas):
    # 2. Construtor Dinâmico de Entradas
    def criar_painel_cargas(qtd):
        md_text = ""
        controles = {}

        for i in range(qtd):
            # Cria chaves únicas para cada carga (ex: no_0, dir_0, mag_0)
            k_no, k_dir, k_mag = f"no_{i}", f"dir_{i}", f"mag_{i}"

            # Cria os controles no dicionário
            controles[k_no] = mo.ui.dropdown(options=opcoes_nos_carga, value="Nó 3")
            controles[k_dir] = mo.ui.dropdown(
                options={"Eixo X": "X", "Eixo Y": "Y"}, value="Eixo Y"
            )
            controles[k_mag] = mo.ui.number(value=1000.0, step=100.0)

            # Monta o texto Markdown chamando as chaves
            md_text += f"**Carga {i + 1}:** &nbsp; Nó: {{{k_no}}} &nbsp;|&nbsp; Direção: {{{k_dir}}} &nbsp;|&nbsp; Magnitude: {{{k_mag}}} N \n\n"

        # Retorna um ÚNICO objeto reativo oficial do Marimo contendo tudo!
        return mo.md(md_text).batch(**controles)


    # Cria o componente perfeitamente reativo
    cargas_reativas = criar_painel_cargas(seletor_qtd_cargas.value)

    # Empilha e mostra na tela
    painel_motor = mo.vstack(
        [
            mo.md("Configuração das cargas"),
            seletor_qtd_cargas,
            cargas_reativas,
        ]
    )
    return cargas_reativas, painel_motor


@app.cell
def _(cargas_reativas, seletor_gdl, sistema):
    if not seletor_gdl.value:
        raise ValueError("Selecione pelo menos um nó para visualizar o gráfico.")

    # ==========================================
    # APLICANDO MÚLTIPLAS CARGAS
    # ==========================================
    sistema.nodal_loads = {}

    # Extrai o mega-dicionário (ex: {'no_0': 3, 'dir_0': 'Y', 'mag_0': 1000.0, 'no_1': ...})
    valores_carga = cargas_reativas.value

    # Descobre quantas cargas existem dividindo por 3 (já que cada carga tem 3 parâmetros)
    qtd_cargas_aplicadas = len(valores_carga) // 3

    for ii in range(qtd_cargas_aplicadas):
        sistema.apply_load(
            node_id=valores_carga[f"no_{ii}"],
            direction=valores_carga[f"dir_{ii}"][-1],  # O [-1] pega a letra X ou Y
            magnitude=valores_carga[f"mag_{ii}"],
        )
    # ==========================================
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Análise Harmônica (Resposta em Frequência)

    Enquanto a análise modal estuda a estrutura isolada, a Análise Harmônica investiga como a estrutura se comporta ao ser submetida a uma carga externa, como o funcionamento de um motor desbalanceado.

    Para que a resposta da estrutura não tenda ao infinito ao atingir a ressonância, é necessário introduzir a dissipação de energia. Fisicamente, utilizamos o modelo de amortecimento proporcional, que assume que a matriz de amortecimento $[C]$ é uma combinação linear da massa e da rigidez do sistema:

    $$[C] = \alpha[M] + \beta[K]$$

    Os coeficientes $\alpha$ e $\beta$ são calibrados para garantir uma taxa de amortecimento crítico (geralmente em torno de $2\%$) ancorada nas duas primeiras frequências naturais mais relevantes para o fenômeno.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Função de Resposta em Frequência (FRF) via Matriz de Impedância

    Diferente da integração no tempo, o solver desenvolvido mapeia a resposta da estrutura operando puramente no domínio da frequência. Substituindo a força harmônica e a solução permanente na equação de movimento, eliminamos a variável "tempo" e geramos a matriz de Impedância Dinâmica Complexa $[H]$:

    $$[H] = [K] - \omega^2[M] + i\omega[C]$$

    O sistema resolvido passa a ser algebricamente simples: $[H]\{U\} = \{F\}$. A parte imaginária ($i\omega[C]$) impõe o atraso de fase na resposta da estrutura (a energia dissipada).

    **A Escala em Decibéis (dB):**
    Escolheu-se representar Função de Resposta em Frequência (FRF) é apresentada na escala logarítmica de decibéis em relação a uma referência ($1\text{ mm}$).

    $$L_{dB} = 20 \log_{10}\left(\frac{A}{A_{ref}}\right)$$
    """)
    return


@app.cell
def _(
    controles_harm,
    freqNaturais,
    mo,
    np,
    opcoes_nos,
    painel_motor,
    plt,
    seletor_faixa_freq,
    seletor_gdl,
    sistema,
):
    f_min, f_max = seletor_faixa_freq.value
    if f_min == f_max:
        f_max += 1

    # Inverte o dicionário para mapear GDL -> Nome
    # Exemplo: Transforma { "Nó 3 (Direção Y)": 10 } em { 10: "Nó 3 (Direção Y)" }
    gdl_para_nome = {valor: chave for chave, valor in opcoes_nos.items()}

    vet_freq = np.arange(f_min, f_max, step=0.1, dtype="float64")
    vet_freq_ops = np.arange(f_min, f_max, step=1, dtype="float64")

    # 2. Calcular Alpha e Beta automaticamente para 2% de amortecimento
    xi = 0.02

    # Converte as duas primeiras frequências do seu solver para rad/s
    w1 = 2 * np.pi * freqNaturais[0]
    w2 = 2 * np.pi * freqNaturais[1]

    # Calcula os parâmetros de Rayleigh
    alpha_calc = 2 * xi * (w1 * w2) / (w1 + w2)
    beta_calc = 2 * xi / (w1 + w2)

    analiseHarmonica = sistema.solve_harmonic_analysis(
        vet_freq, alpha_rayleigh=alpha_calc, beta_rayleigh=beta_calc
    )

    # 1. Cria DOIS subplots empilhados (2 linhas, 1 coluna), compartilhando o eixo X
    fig2, (ax_mag, ax_fase) = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 8), sharex=True
    )

    for gdl in seletor_gdl.value:
        # O resultado bruto é um número complexo
        resposta_complexa = analiseHarmonica[gdl, :]

        # ==========================================
        # CÁLCULO DA MAGNITUDE (dB)
        # ==========================================
        amplitude_m = np.abs(resposta_complexa)
        A_ref = 1e-3
        amplitude_m_segura = np.maximum(amplitude_m, 1e-12)
        amplitude_dB = 20 * np.log10(amplitude_m_segura / A_ref)

        # ==========================================
        # CÁLCULO DA FASE (Graus)
        # ==========================================

        fase_rad = np.angle(resposta_complexa)

        # A mágica acontece aqui: desenrola os saltos de 2*pi
        fase_rad_continua = np.unwrap(fase_rad)

        nome_da_curva = gdl_para_nome.get(gdl, f"GDL {gdl}")

        # Plota nos respectivos eixos
        ax_mag.plot(vet_freq, amplitude_dB, label=nome_da_curva)
        ax_fase.plot(vet_freq, fase_rad_continua, label=nome_da_curva)

    # ==========================================
    # FORMATAÇÃO DO GRÁFICO DE MAGNITUDE
    # ==========================================
    ax_mag.set_title("Resposta Harmônica (Diagrama de Bode)")
    ax_mag.set_ylabel("Amplitude [dB ref 1 mm]")
    ax_mag.grid(True, which="both", linestyle=":", alpha=0.7)
    ax_mag.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

    # ==========================================
    # FORMATAÇÃO DO GRÁFICO DE FASE
    # ==========================================
    ax_fase.set_xlabel("Frequência de Excitação [Hz]")
    ax_fase.set_ylabel("Fase [rad]")
    # Força o eixo Y a ter marcações redondas de 90 em 90 graus
    ax_fase.grid(True, linestyle=":", alpha=0.7)
    ax_fase.legend()

    # Ajusta os espaçamentos para os textos não se sobreporem
    fig2.tight_layout()

    # fig2.savefig(
    #     "docs_trelica/docs/assets/comparativo_frf.png", dpi=400, bbox_inches="tight"
    # )

    mo.vstack(
        [
            painel_motor,
            controles_harm,
            mo.as_html(fig2),
            mo.md(f"Alpha e Beta calculados {alpha_calc:.4g}, {beta_calc:.4g}"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
