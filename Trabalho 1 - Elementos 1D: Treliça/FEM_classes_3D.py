"""
Implementação 3D (3 GDL por nó) de elementos de treliça para MEF
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, spsolve


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

    def assemble_stiffness(self) -> csr_matrix[np.float64]:
        """
        Monta a matriz de rigidez global e aplica as condições de contorno.

        Retorna:
            scipy.sparse.csr_matrix: Matriz de rigidez global montada.
        """
        # Inicializa a matriz global usando o formato LIL para construção eficiente
        K_global: lil_matrix = lil_matrix[np.float64](
            (self.total_dofs, self.total_dofs)
        )

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

    def assemble_mass(self) -> csr_matrix[np.float64]:
        """
        Monta a matriz de massa global e aplica as condições de contorno.

        Retorna:
            scipy.sparse.csr_matrix: Matriz de massa global montada.
        """
        M_global: lil_matrix[np.float64] = lil_matrix[np.float64](
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
