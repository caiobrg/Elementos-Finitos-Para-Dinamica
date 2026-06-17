import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import gmsh
    import marimo as mo
    import matplotlib.pyplot as plt
    import scienceplots
    import numpy as np

    # Imports do projeto
    from FEM_classes_3D import CableElement3D, FEMSystem, FrameElement3D, Node
    from plot_functions import plot_mode_shape

    plt.style.use("science")
    return (
        CableElement3D,
        FEMSystem,
        FrameElement3D,
        Node,
        gmsh,
        mo,
        np,
        plot_mode_shape,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Desafio 1: Ponte Estaiada com Vibração Induzida por Vento

    Este documento apresenta a modelagem paramétrica, discretização e análise dinâmica (modal e harmônica) de uma ponte estaiada sob a ação de vento lateral.

    ---

    ## Respostas às Questões do Desafio

    **1. Qual(is) elemento(s) finito(s) você usa para o tabuleiro? Justifique.**
    Utiliza-se o **Elemento de Pórtico Plano** (Viga bidimensional com 3 GDLs por nó). O tabuleiro está sujeito a flexão no plano vertical e vibração longitudinal, exigindo um elemento capaz de capturar esforços axiais e momentos fletores combinados.

    **2. Qual(is) elemento(s) para as torres?**
    Utiliza-se o **Elemento de Pórtico Plano**. As torres de 40 m estão engastadas na base  e sofrem flexão e compressão simultâneas devido à ação dos cabos e do vento, necessitando de continuidade rotacional para transmissão de momento fletor.

    **3. Qual(is) elemento(s) para os cabos? É necessário considerar não linearidade geométrica?**
    Utiliza-se o **Elemento de Cabo com não linearidade geométrica**. Sim, a não linearidade é estritamente necessária. Os cabos possuem tração inicial de 500 kN. Como não possuem rigidez elástica à flexão, a resistência à vibração transversal advém puramente dessa pré-tração (matriz de rigidez geométrica). Ignorar isso resultaria em frequências nulas.

    **4. Como você conecta os cabos ao tabuleiro e às torres? (rótula ou engaste?)**
    A conexão deve ser **rotulada** (articulada). Os cabos trabalham exclusivamente sob tração. O elemento de cabo foi formulado possuindo apenas GDLs translacionais. Isso garante matematicamente uma rótula perfeita na matriz global, não transmitindo momentos fletores espúrios para as torres e o tabuleiro.

    **5. E se você tentasse modelar os cabos como elementos de barra comum? Quais erros isso traria para as frequências naturais?**
    O elemento de barra comum possui apenas rigidez axial elástica. Na matriz global, a rigidez transversal seria considerada zero. Isso subestimaria drasticamente (ou zeraria) as frequências transversais locais dos cabos e impossibilitaria a captura correta do acoplamento dinâmico global da ponte sob ação do vento lateral.
    """)
    return


@app.cell
def _(CableElement3D, FEMSystem, FrameElement3D, Node, gmsh, np):
    def create_mesh_and_solve():
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("Ponte_Estaiada_Desafio1")

        # 1. Parâmetros Físicos e Geométricos
        # Tabuleiro (Concreto)
        E_c = 30e9
        rho_c = 2500.0
        nu_c = 0.2
        G_c = E_c / (2 * (1 + nu_c))
        A_deck = 10.0
        Iy_deck = 50.0
        Iz_deck = 10.0
        J_deck = 40.0

        # Torres (Aço)
        E_s = 200e9
        rho_s = 7850.0
        nu_s = 0.3
        G_s = E_s / (2 * (1 + nu_s))
        A_tower = 0.5
        Iy_tower = 0.2
        Iz_tower = 0.2
        J_tower = 0.4

        # Cabos (Aço)
        d_cabo = 0.050
        A_cable = np.pi * (d_cabo**2) / 4.0
        T_inicial = 500e3

        # 2. Definição da Geometria (Gmsh)
        lc = 2.0  # Tamanho característico da malha

        # Nós do tabuleiro
        pt_deck_start = gmsh.model.geo.addPoint(0, 0, 0, lc)
        pt_tower1_base = gmsh.model.geo.addPoint(50, 0, 0, lc)
        pt_anc1 = gmsh.model.geo.addPoint(75, 0, 0, lc)
        pt_anc2 = gmsh.model.geo.addPoint(100, 0, 0, lc)
        pt_anc3 = gmsh.model.geo.addPoint(125, 0, 0, lc)
        pt_tower2_base = gmsh.model.geo.addPoint(150, 0, 0, lc)
        pt_deck_end = gmsh.model.geo.addPoint(200, 0, 0, lc)

        # Nós das torres
        pt_tower1_top = gmsh.model.geo.addPoint(50, 40, 0, lc)
        pt_tower2_top = gmsh.model.geo.addPoint(150, 40, 0, lc)

        # Linhas: Tabuleiro, Torres e Cabos
        deck_lines = [
            gmsh.model.geo.addLine(pt_deck_start, pt_tower1_base),
            gmsh.model.geo.addLine(pt_tower1_base, pt_anc1),
            gmsh.model.geo.addLine(pt_anc1, pt_anc2),
            gmsh.model.geo.addLine(pt_anc2, pt_anc3),
            gmsh.model.geo.addLine(pt_anc3, pt_tower2_base),
            gmsh.model.geo.addLine(pt_tower2_base, pt_deck_end),
        ]

        tower_lines = [
            gmsh.model.geo.addLine(pt_tower1_base, pt_tower1_top),
            gmsh.model.geo.addLine(pt_tower2_base, pt_tower2_top),
        ]

        cable_lines = [
            gmsh.model.geo.addLine(pt_tower1_top, pt_anc1),
            gmsh.model.geo.addLine(pt_tower1_top, pt_anc2),
            gmsh.model.geo.addLine(pt_tower1_top, pt_anc3),
            gmsh.model.geo.addLine(pt_tower2_top, pt_anc1),
            gmsh.model.geo.addLine(pt_tower2_top, pt_anc2),
            gmsh.model.geo.addLine(pt_tower2_top, pt_anc3),
        ]

        gmsh.model.geo.synchronize()

        # 3. Discretização e Physical Groups
        PG_DECK, PG_TOWER, PG_CABLE, PG_SUPPORT = 1, 2, 3, 4

        gmsh.model.addPhysicalGroup(1, deck_lines, PG_DECK)
        gmsh.model.addPhysicalGroup(1, tower_lines, PG_TOWER)
        gmsh.model.addPhysicalGroup(1, cable_lines, PG_CABLE)
        gmsh.model.addPhysicalGroup(
            0, [pt_deck_start, pt_tower1_base, pt_tower2_base, pt_deck_end], PG_SUPPORT
        )

        gmsh.model.mesh.generate(1)

        # 4. Extração e Instanciação (Gmsh -> MEF Python)
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        my_nodes = {}
        fem_nodes_list = []

        for i, tag in enumerate(nodeTags):
            n = Node(
                id=i,
                x=nodeCoords[3 * i],
                y=nodeCoords[3 * i + 1],
                z=nodeCoords[3 * i + 2],
            )
            my_nodes[tag] = n
            fem_nodes_list.append(n)

        fem_elements_list = []
        element_id_counter = 0

        def extract_elements(physical_group_tag, elem_type_handler):
            nonlocal element_id_counter
            entities = gmsh.model.getEntitiesForPhysicalGroup(1, physical_group_tag)
            for e in entities:
                elemTypes, _, elemNodeTags = gmsh.model.mesh.getElements(1, e)
                if not elemTypes:
                    continue

                nodes_flat = elemNodeTags[0]
                for i in range(0, len(nodes_flat), 2):
                    n1, n2 = my_nodes[nodes_flat[i]], my_nodes[nodes_flat[i + 1]]
                    fem_elements_list.append(
                        elem_type_handler(element_id_counter, n1, n2)
                    )
                    element_id_counter += 1

        extract_elements(
            PG_DECK,
            lambda eid, n1, n2: FrameElement3D(
                eid, n1, n2, E_c, G_c, A_deck, Iy_deck, Iz_deck, J_deck, rho_c
            ),
        )
        extract_elements(
            PG_TOWER,
            lambda eid, n1, n2: FrameElement3D(
                eid, n1, n2, E_s, G_s, A_tower, Iy_tower, Iz_tower, J_tower, rho_s
            ),
        )
        extract_elements(
            PG_CABLE,
            lambda eid, n1, n2: CableElement3D(
                eid, n1, n2, E_s, A_cable, rho_s, T_inicial
            ),
        )

        # 5. Condições de Contorno
        boundary_dofs = []

        # Bloqueio de GDLs fora do plano (simulação de pórtico 2D)
        for no in fem_nodes_list:
            boundary_dofs.extend([no.dof_uz, no.dof_rx, no.dof_ry])

        # Engastes da base
        support_node_tags = gmsh.model.mesh.getNodesForPhysicalGroup(0, PG_SUPPORT)[0]
        for tag in support_node_tags:
            boundary_dofs.extend(
                [my_nodes[tag].dof_ux, my_nodes[tag].dof_uy, my_nodes[tag].dof_rz]
            )

        gmsh.finalize()

        bridge_system = FEMSystem(
            nodes=fem_nodes_list, elements=fem_elements_list, boundarycond=boundary_dofs
        )

        # 6. Carregamento do Vento e Análise Harmônica
        V_vento = 25.0
        rho_ar = 1.2
        Cd_deck, D_deck = 1.3, 3.5
        Cd_tower, D_tower = 1.2, 2.0

        qw_deck = 0.5 * rho_ar * (V_vento**2) * Cd_deck * D_deck
        qw_tower = 0.5 * rho_ar * (V_vento**2) * Cd_tower * D_tower

        # Aplicação de forças nodais equivalentes baseadas nos grupos
        for el in fem_elements_list:
            if el.rho == rho_c:
                f_nodal = (qw_deck * el.L) / 2.0
                bridge_system.apply_load(el.node_i.id, "Y", f_nodal)
                bridge_system.apply_load(el.node_j.id, "Y", f_nodal)
            elif el.rho == rho_s and hasattr(el, "G"):
                f_nodal = (qw_tower * el.L) / 2.0
                bridge_system.apply_load(el.node_i.id, "X", f_nodal)
                bridge_system.apply_load(el.node_j.id, "X", f_nodal)

        # Resolução dos sistemas
        faixa_freqs = np.arange(0.01, 10.0, 0.01)
        respostas_complexas = bridge_system.solve_harmonic_analysis(
            freq_range_hz=faixa_freqs, alpha_rayleigh=0.05, beta_rayleigh=0.002
        )
        amplitudes = np.abs(respostas_complexas)
        frequencias, formas = bridge_system.solve_modal_analysis(num_modes=50)

        return bridge_system, frequencias, formas, amplitudes, faixa_freqs

    return (create_mesh_and_solve,)


@app.cell
def _(create_mesh_and_solve, mo):
    with mo.status.spinner("Montando malha e resolvendo análise modal..."):
        sys_mef, freqs, modos, amplitudes, faixa_freqs = create_mesh_and_solve()

    mo.md(
        f"### Modelo resolvido com sucesso!\nNós: {len(sys_mef.nodes)} | Elementos: {len(sys_mef.elements)}"
    )
    return amplitudes, faixa_freqs, freqs, modos, sys_mef


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Análise Modal

    Abaixo, exploramos as frequências naturais da estrutura. Nota-se que as primeiras frequências estão na faixa de 1 a 2 Hz, caracterizando a vibração puramente transversal dos cabos sob o efeito da pré-tração (modos locais). Modos de maior energia e frequência exibem o acoplamento estrutural global (flexão do tabuleiro). A simetria geométrica da ponte gera autovalores duplicados para as respostas dos cabos.
    """)
    return


@app.cell
def _(freqs, mo):
    f_rows = "".join(
        f"<tr><td><b>Modo {i + 1}</b></td><td>{f:.3f} Hz</td></tr>"
        for i, f in enumerate(freqs)
    )
    mo.md(
        f"""
        <details>
        <summary><b>Clique para expandir a tabela de Frequências Naturais</b></summary>
        <table>
            <thead>
                <tr>
                    <th style="text-align: left; padding-right: 20px;">Modo</th>
                    <th style="text-align: left;">Frequência Natural</th>
                </tr>
            </thead>
            <tbody>
                {f_rows}
            </tbody>
        </table>
        </details>
        """
    )
    return


@app.cell
def _(mo):
    mode_select = mo.ui.dropdown(
        options=[i for i in range(50)],
        value=0,
        label="Selecione o Modo de Vibração",
    )

    scale_select = mo.ui.slider(
        start=100,
        stop=15000,
        step=100,
        value=500,
        label="Escala de Deformação (Visualização)",
        debounce=True,
    )
    return mode_select, scale_select


@app.cell
def _(
    freqs,
    mo,
    mode_select,
    modos,
    plot_mode_shape,
    plt,
    scale_select,
    sys_mef,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_mode_shape(
        system=sys_mef,
        frequencies=freqs,
        eigenvectors=modos,
        axs=ax,
        mode_index=mode_select.value,
        scale=scale_select.value,
        use_hue=True,
        plot_original=True,
    )

    plt.tight_layout()

    mo.vstack(
        [mo.hstack([mode_select, scale_select], justify="start"), mo.as_html(fig)]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Modelagem do Forçamento pelo Vento e Análise Harmônica

    A simulação da ação do vento na estrutura foi desenvolvida em três etapas analíticas, conectando a aerodinâmica clássica à resolução matricial do Método dos Elementos Finitos (MEF) no domínio da frequência:

    **1. Formulação da Carga Aerodinâmica Distribuída**
    O vento atua exercendo pressão sobre as áreas expostas da ponte. A força estática equivalente por unidade de comprimento ($q_w$, em N/m) atuante nos perfis foi calculada pela equação de arrasto:

    $$q_w = \frac{1}{2} \rho_{ar} V_w^2 C_d D$$

    Para este cenário, assumiu-se a densidade do ar $\rho_{ar} = 1.2 \text{ kg/m}^3$ e velocidade constante do vento $V_w = 25 \text{ m/s}$. Os coeficientes de arrasto ($C_d$) e dimensões características ($D$) foram particularizados para a geometria de cada seção (seção caixão para o tabuleiro e seção tubular para as torres).

    **2. Conversão para Forças Nodais Equivalentes**
    Para incorporar o carregamento distribuído de fluido no modelo discreto do MEF, aplicou-se o princípio das forças nodais equivalentes. A força contínua ao longo de cada elemento finito linear ($q_w \cdot L$) foi dividida simetricamente em forças pontuais aplicadas aos nós de conectividade da malha.
    * **Tabuleiro:** O vento incidente gera forças verticais devido à assimetria de escoamento e desprendimento de vórtices. Logo, as cargas foram aplicadas no grau de liberdade vertical ($Y$).
    * **Torres:** Estão sujeitas ao arrasto direto do vento, resultando em forças concentradas no grau de liberdade horizontal transversal ($X$).

    **3. Varredura Harmônica no Domínio da Frequência**
    Com o vetor de forças globais $\{F\}$ estabelecido, o comportamento em ressonância foi avaliado por meio de uma varredura de frequências (0.01 Hz a 10 Hz). Para cada passo de excitação $\omega$, montou-se a matriz de Impedância Dinâmica do sistema:

    $$[H(\omega)] = [K] - \omega^2[M] + i\omega[C]$$

    Onde $[K]$ é a matriz de rigidez (incluindo o efeito geométrico dos cabos tracionados), $[M]$ é a matriz de massa e $[C]$ é a matriz de amortecimento de Rayleigh, introduzida para modelar a dissipação de energia e limitar fisicamente as amplitudes de pico.

    O gráfico abaixo apresenta a Função de Resposta em Frequência (FRF) resultante da translação vertical do nó central do vão. É possível observar nitidamente que as frequências de ressonância (picos de deslocamento) coincidem com os modos globais de vibração obtidos na análise modal anterior.

    É importante notar que o vento não foi modelado como um processo estocástico, esta é um simplificação que permite a construção da FRF sem que seja necessário a integração temporal do problema.
    """)
    return


@app.cell
def _(amplitudes, faixa_freqs, np, plt, sys_mef):
    no_centro = next(
        no for no in sys_mef.nodes if np.isclose(no.x, 100.0) and np.isclose(no.y, 0.0)
    )
    gdl_uy_centro = no_centro.dof_uy

    deslocamentos_centro = amplitudes[gdl_uy_centro, :]

    plt.figure(figsize=(10, 5))
    plt.plot(
        faixa_freqs,
        deslocamentos_centro,
        linewidth=2,
        label="Nó Central do Tabuleiro (Y)",
    )
    plt.title("Resposta Harmônica da Ponte Estaiada sob Ação do Vento")
    plt.xlabel("Frequência de Excitação do Vento [Hz]")
    plt.ylabel("Amplitude do Deslocamento [m]")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
