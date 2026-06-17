"""
Implementação 3D (3 GDL por nó) de elementos 1D para MEF
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh, spsolve


@dataclass
class Node:
    """
    Representa um nó em uma estrutura de treliça 2D.

    Atributos:
        id (int): Identificador único do nó.
        x (float): Coordenada X do nó.
        y (float): Coordenada Y do nó.
        z (float): Coordenada Z do nó.
        dof_ux (int): Índice global do grau de liberdade na direção X.
        dof_uy (int): Índice global do grau de liberdade na direção Y.
        dof_uz (int): Índice global do grau de liberdade na direção Z.
    """

    id: int
    x: float
    y: float
    z: float

    # Índices dos GDLs (Mapeamento para as matrizes globais)
    dof_ux: int = field(init=False)
    dof_uy: int = field(init=False)
    dof_uz: int = field(init=False)
    dof_rx: int = field(init=False)
    dof_ry: int = field(init=False)
    dof_rz: int = field(init=False)

    # Resultados Físicos (Valores calculados de deslocamento e rotação)
    # Inicializados com 0s para serem alocados depois
    res_ux: float = 0.0
    res_uy: float = 0.0
    res_uz: float = 0.0
    res_rx: float = 0.0
    res_ry: float = 0.0
    res_rz: float = 0.0

    def __post_init__(self):
        # Mapeamento dos índices dos GDLs
        self.dof_ux = 6 * self.id
        self.dof_uy = 6 * self.id + 1
        self.dof_uz = 6 * self.id + 2
        self.dof_rx = 6 * self.id + 3
        self.dof_ry = 6 * self.id + 4
        self.dof_rz = 6 * self.id + 5


class Element1D(ABC):
    """
    Classe base abstrata para todos os elementos unidimensionais (Treliça, Viga, Pórtico).
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

        # Cálculos geométricos comuns a todos os elementos 1D
        dx: float = node_j.x - node_i.x
        dy: float = node_j.y - node_i.y
        dz: float = node_j.z - node_i.z

        self.L: float = np.sqrt(dx**2 + dy**2 + dz**2)

        if self.L == 0:
            raise ValueError(f"Elemento {self.id} tem comprimento zero.")

        # Cossenos diretores (Vetor de orientação)
        self.v: npt.NDArray[np.float64] = np.array(
            [dx / self.L, dy / self.L, dz / self.L]
        )

        # Flags e mapeamentos que as classes filhas deverão preencher
        self.dofs: list[int] = []

    @abstractmethod
    def get_stiffness_matrix(self) -> npt.NDArray[np.float64]:
        """Deve ser implementado pelas classes filhas."""
        pass

    @abstractmethod
    def get_mass_matrix(self) -> npt.NDArray[np.float64]:
        """Deve ser implementado pelas classes filhas."""
        pass


class TrussElement(Element1D):
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
        node_i: "Node",
        node_j: "Node",
        E: float,
        A: float,
        rho: float,
    ) -> None:
        # Chama a inicialização da classe mãe (calcula L, v, etc.)
        super().__init__(id, node_i, node_j, E, A, rho)

        # A treliça só usa 6 GDLs (apenas translações) na sua matriz local
        self.dofs: list[int] = [
            node_i.dof_ux,
            node_i.dof_uy,
            node_i.dof_uz,
            node_j.dof_ux,
            node_j.dof_uy,
            node_j.dof_uz,
        ]

    @override
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

    @override
    def get_mass_matrix(self) -> npt.NDArray[np.float64]:
        coef: float = (self.rho * self.A * self.L) / 6
        I3 = np.eye(3)  # Matriz Identidade 3x3

        # A matriz de massa consistente em 3D é distribuída simetricamente
        M_e_bar = coef * np.block([[2 * I3, I3], [I3, 2 * I3]])
        return M_e_bar


class CableElement3D(Element1D):
    """
    Representa um elemento de cabo tensionado no espaço 3D.
    Inclui não linearidade geométrica (rigidez transversal devido à pré-tração).
    Não possui rigidez à flexão ou torção.
    """

    def __init__(
        self,
        id: int,
        node_i: "Node",
        node_j: "Node",
        E: float,
        A: float,
        rho: float,
        T_initial: float,  # Novo parâmetro obrigatório: Tração Inicial (Positivo = Tração)
    ) -> None:
        super().__init__(id, node_i, node_j, E, A, rho)
        self.T_initial: float = T_initial

        # O cabo também só usa 6 GDLs (apenas translações)
        self.dofs: list[int] = [
            node_i.dof_ux,
            node_i.dof_uy,
            node_i.dof_uz,
            node_j.dof_ux,
            node_j.dof_uy,
            node_j.dof_uz,
        ]

    @override
    def get_stiffness_matrix(self) -> npt.NDArray[np.float64]:
        # 1. Matriz de Rigidez Elástica [K_e] (Comportamento de Treliça)
        coef_e: float = (self.E * self.A) / self.L

        # Produto externo do vetor v cria a matriz de projeção axial 3x3
        Lambda: npt.NDArray[np.float64] = np.outer(self.v, self.v)

        K_e: npt.NDArray[np.float64] = coef_e * np.block(
            [[Lambda, -Lambda], [-Lambda, Lambda]]
        )

        # 2. Matriz de Rigidez Geométrica [K_g] (Efeito da Pré-Tração)
        coef_g: float = self.T_initial / self.L

        I3 = np.eye(3)
        # A rigidez atua no plano transversal (I3 - Lambda)
        Proj_transverse: npt.NDArray[np.float64] = I3 - Lambda

        K_g: npt.NDArray[np.float64] = coef_g * np.block(
            [[Proj_transverse, -Proj_transverse], [-Proj_transverse, Proj_transverse]]
        )

        # Matriz de rigidez total do cabo
        return K_e + K_g

    @override
    def get_mass_matrix(self) -> npt.NDArray[np.float64]:
        # A matriz de massa de um cabo tracionado é idêntica à da treliça
        coef: float = (self.rho * self.A * self.L) / 6
        I3 = np.eye(3)

        M_e_cable = coef * np.block([[2 * I3, I3], [I3, 2 * I3]])
        return M_e_cable


class BeamElement3D(Element1D):
    """
    Representa um elemento de viga pura no espaço 3D.
    Suporta APENAS flexão em dois planos.
    NÃO possui rigidez axial nem à torção.
    """

    def __init__(
        self,
        id: int,
        node_i: "Node",
        node_j: "Node",
        E: float,
        Iy: float,
        Iz: float,
        A: float,  # Necessário para calcular a massa
        rho: float,
    ) -> None:
        super().__init__(id, node_i, node_j, E, A, rho)
        self.Iy: float = Iy
        self.Iz: float = Iz

        # Usa todos os 12 GDLs para a transformação geométrica no espaço
        self.dofs: list[int] = [
            node_i.dof_ux,
            node_i.dof_uy,
            node_i.dof_uz,
            node_i.dof_rx,
            node_i.dof_ry,
            node_i.dof_rz,
            node_j.dof_ux,
            node_j.dof_uy,
            node_j.dof_uz,
            node_j.dof_rx,
            node_j.dof_ry,
            node_j.dof_rz,
        ]

    def get_transformation_matrix(self) -> npt.NDArray[np.float64]:
        """Calcula a matriz de transformação 12x12 baseada em cossenos diretores."""
        vx = self.v

        v_up = np.array([0.0, 0.0, 1.0])
        if np.allclose(np.abs(vx), [0.0, 0.0, 1.0]):
            v_up = np.array([0.0, 1.0, 0.0])

        vz = np.cross(vx, v_up)
        vz /= np.linalg.norm(vz)
        vy = np.cross(vz, vx)

        l_mat = np.vstack([vx, vy, vz])

        T = np.zeros((12, 12))
        for i in range(4):
            T[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = l_mat
        return T

    def get_stiffness_matrix_local(self) -> npt.NDArray[np.float64]:
        """Constrói a matriz de rigidez 12x12 (apenas termos de flexão)."""
        L, E, Iy, Iz = self.L, self.E, self.Iy, self.Iz
        k = np.zeros((12, 12))

        # OBS: Linhas/colunas 0 e 6 (axial) e 3 e 9 (torção) permanecem com ZERO absoluto.

        # Flexão no plano xy (envolve uy e rz)
        k_xy = np.array(
            [
                [
                    12 * E * Iz / L**3,
                    6 * E * Iz / L**2,
                    -12 * E * Iz / L**3,
                    6 * E * Iz / L**2,
                ],
                [
                    6 * E * Iz / L**2,
                    4 * E * Iz / L,
                    -6 * E * Iz / L**2,
                    2 * E * Iz / L,
                ],
                [
                    -12 * E * Iz / L**3,
                    -6 * E * Iz / L**2,
                    12 * E * Iz / L**3,
                    -6 * E * Iz / L**2,
                ],
                [
                    6 * E * Iz / L**2,
                    2 * E * Iz / L,
                    -6 * E * Iz / L**2,
                    4 * E * Iz / L,
                ],
            ]
        )
        idx_xy = [1, 5, 7, 11]
        k[np.ix_(idx_xy, idx_xy)] += k_xy

        # Flexão no plano xz (envolve uz e ry)
        k_xz = np.array(
            [
                [
                    12 * E * Iy / L**3,
                    -6 * E * Iy / L**2,
                    -12 * E * Iy / L**3,
                    -6 * E * Iy / L**2,
                ],
                [
                    -6 * E * Iy / L**2,
                    4 * E * Iy / L,
                    6 * E * Iy / L**2,
                    2 * E * Iy / L,
                ],
                [
                    -12 * E * Iy / L**3,
                    6 * E * Iy / L**2,
                    12 * E * Iy / L**3,
                    6 * E * Iy / L**2,
                ],
                [
                    -6 * E * Iy / L**2,
                    2 * E * Iy / L,
                    6 * E * Iy / L**2,
                    4 * E * Iy / L,
                ],
            ]
        )
        idx_xz = [2, 4, 8, 10]
        k[np.ix_(idx_xz, idx_xz)] += k_xz

        return k

    @override
    def get_stiffness_matrix(self) -> npt.NDArray[np.float64]:
        """Transforma a matriz local para o sistema global."""
        kl = self.get_stiffness_matrix_local()
        T = self.get_transformation_matrix()
        return T.T @ kl @ T

    def get_mass_matrix_local(self) -> npt.NDArray[np.float64]:
        """Constrói a matriz de massa consistente 12x12."""
        L, rho, A = self.L, self.rho, self.A
        m = np.zeros((12, 12))
        coef = (rho * A * L) / 420.0

        # Massa consistente para flexão no plano XY (envolve uy e rz)
        m_xy = coef * np.array(
            [
                [156, 22 * L, 54, -13 * L],
                [22 * L, 4 * L**2, 13 * L, -3 * L**2],
                [54, 13 * L, 156, -22 * L],
                [-13 * L, -3 * L**2, -22 * L, 4 * L**2],
            ]
        )
        idx_xy = [1, 5, 7, 11]
        m[np.ix_(idx_xy, idx_xy)] += m_xy

        # Massa consistente para flexão no plano XZ (envolve uz e ry)
        # Nota: Os sinais dos termos lineares que cruzam com rotação invertem
        # devido à convenção da regra da mão direita no eixo Y.
        m_xz = coef * np.array(
            [
                [156, -22 * L, 54, 13 * L],
                [-22 * L, 4 * L**2, -13 * L, -3 * L**2],
                [54, -13 * L, 156, 22 * L],
                [13 * L, -3 * L**2, 22 * L, 4 * L**2],
            ]
        )
        idx_xz = [2, 4, 8, 10]
        m[np.ix_(idx_xz, idx_xz)] += m_xz

        # Massa para GDLs axiais (mesmo sem rigidez associada nesta viga simples)
        m_axial = (rho * A * L / 6.0) * np.array([[2, 1], [1, 2]])
        m[np.ix_([0, 6], [0, 6])] += m_axial

        return m

    @override
    def get_mass_matrix(self) -> npt.NDArray[np.float64]:
        """Transforma a matriz de massa local para o sistema global."""
        ml = self.get_mass_matrix_local()
        T = self.get_transformation_matrix()
        return T.T @ ml @ T


class FrameElement3D(Element1D):
    """
    Representa um elemento de pórtico espacial (frame) em 3D.
    Suporta esforços axiais, torção e flexão em dois planos.
    """

    def __init__(
        self,
        id: int,
        node_i: Node,
        node_j: Node,
        E: float,
        G: float,
        A: float,
        Iy: float,
        Iz: float,
        J: float,
        rho: float,
    ) -> None:
        super().__init__(id, node_i, node_j, E, A, rho)
        self.G: float = G
        self.Iy: float = Iy
        self.Iz: float = Iz
        self.J: float = J

        # Mapeamento dos 12 GDLs (6 por nó)
        self.dofs: list[int] = [
            node_i.dof_ux,
            node_i.dof_uy,
            node_i.dof_uz,
            node_i.dof_rx,
            node_i.dof_ry,
            node_i.dof_rz,
            node_j.dof_ux,
            node_j.dof_uy,
            node_j.dof_uz,
            node_j.dof_rx,
            node_j.dof_ry,
            node_j.dof_rz,
        ]

    def get_transformation_matrix(self) -> npt.NDArray[np.float64]:
        """Calcula a matriz de transformação 12x12 baseada em cossenos diretores."""
        vx = self.v  # Eixo local x (unitário)

        # Vetor auxiliar para definir a orientação 'vertical' da secção
        v_up = np.array([0.0, 0.0, 1.0])
        if np.allclose(np.abs(vx), [0.0, 0.0, 1.0]):
            v_up = np.array([0.0, 1.0, 0.0])

        vz = np.cross(vx, v_up)
        vz /= np.linalg.norm(vz)
        vy = np.cross(vz, vx)

        # Matriz de rotação 3x3
        l_mat = np.vstack([vx, vy, vz])

        # Expansão para 12x12
        T = np.zeros((12, 12))
        for i in range(4):
            T[i * 3 : (i + 1) * 3, i * 3 : (i + 1) * 3] = l_mat
        return T

    def get_stiffness_matrix_local(self) -> npt.NDArray[np.float64]:
        """Constrói a matriz de rigidez 12x12 no sistema de coordenadas local."""
        L, E, G, A, Iy, Iz, J = (
            self.L,
            self.E,
            self.G,
            self.A,
            self.Iy,
            self.Iz,
            self.J,
        )
        k = np.zeros((12, 12))

        # Rigidez Axial (x)
        axial = E * A / L
        k[0, 0] = k[6, 6] = axial
        k[0, 6] = k[6, 0] = -axial

        # Rigidez à Torção (rx)
        torsion = G * J / L
        k[3, 3] = k[9, 9] = torsion
        k[3, 9] = k[9, 3] = -torsion

        # Flexão no plano xy (envolve uy e rz)
        k_xy = np.array(
            [
                [
                    12 * E * Iz / L**3,
                    6 * E * Iz / L**2,
                    -12 * E * Iz / L**3,
                    6 * E * Iz / L**2,
                ],
                [
                    6 * E * Iz / L**2,
                    4 * E * Iz / L,
                    -6 * E * Iz / L**2,
                    2 * E * Iz / L,
                ],
                [
                    -12 * E * Iz / L**3,
                    -6 * E * Iz / L**2,
                    12 * E * Iz / L**3,
                    -6 * E * Iz / L**2,
                ],
                [
                    6 * E * Iz / L**2,
                    2 * E * Iz / L,
                    -6 * E * Iz / L**2,
                    4 * E * Iz / L,
                ],
            ]
        )
        idx_xy = [1, 5, 7, 11]
        k[np.ix_(idx_xy, idx_xy)] += k_xy

        # Flexão no plano xz (envolve uz e ry)
        # Nota: Sinais invertidos em alguns termos devido à convenção de mão direita em y
        k_xz = np.array(
            [
                [
                    12 * E * Iy / L**3,
                    -6 * E * Iy / L**2,
                    -12 * E * Iy / L**3,
                    -6 * E * Iy / L**2,
                ],
                [
                    -6 * E * Iy / L**2,
                    4 * E * Iy / L,
                    6 * E * Iy / L**2,
                    2 * E * Iy / L,
                ],
                [
                    -12 * E * Iy / L**3,
                    6 * E * Iy / L**2,
                    12 * E * Iy / L**3,
                    6 * E * Iy / L**2,
                ],
                [
                    -6 * E * Iy / L**2,
                    2 * E * Iy / L,
                    6 * E * Iy / L**2,
                    4 * E * Iy / L,
                ],
            ]
        )
        idx_xz = [2, 4, 8, 10]
        k[np.ix_(idx_xz, idx_xz)] += k_xz

        return k

    @override
    def get_stiffness_matrix(self) -> npt.NDArray[np.float64]:
        """Transforma a matriz local para o sistema global."""
        kl = self.get_stiffness_matrix_local()
        T = self.get_transformation_matrix()
        return T.T @ kl @ T

    @override
    def get_mass_matrix(self) -> npt.NDArray[np.float64]:
        """Implementação simplificada da matriz de massa consistente 12x12."""
        # Para fins didáticos, pode-se expandir a lógica de [cite: 1036]
        # ou usar massa concentrada (lumped) para validação inicial.
        L, rho, A = self.L, self.rho, self.A
        m_total = rho * A * L
        # Exemplo de matriz diagonal simples (lumped mass) para teste:
        return np.eye(12) * (m_total / 2.0)


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
        elements: list[Element1D],
        boundarycond: list[int] | None = None,
    ) -> None:
        self.nodes: list[Node] = nodes
        self.elements: list[Element1D] = elements
        self.boundarycond: list[int] = boundarycond if boundarycond is not None else []
        self.total_dofs: int = len(nodes) * 6

        self._update_free_dofs()

        # Dicionário para armazenar as forças {grau_de_liberdade: magnitude}
        self.nodal_loads: dict[int, float] = {}

    def _update_free_dofs(self) -> None:
        """Atualiza o array de GDLs livres baseado nas condições de contorno."""
        all_dofs: npt.NDArray[np.int32] = np.arange(self.total_dofs, dtype="int32")
        if self.boundarycond:
            # Remove duplicatas e ordena para garantir consistência
            unique_bc = sorted(list(set(self.boundarycond)))
            self.free_dofs_array: npt.NDArray[np.int32] = np.delete(all_dofs, unique_bc)
        else:
            self.free_dofs_array = all_dofs

    def _auto_constrain_inactive_dofs(self) -> None:
        """
        Verifica a diagonal principal da matriz de rigidez e trava
        automaticamente os graus de liberdade (GDLs) sem rigidez associada.
        """
        # Vetor para acumular a rigidez de cada GDL (apenas a diagonal)
        diag_K = np.zeros(self.total_dofs)

        # Varre os elementos e soma as contribuições locais na diagonal global
        for el in self.elements:
            K_e = el.get_stiffness_matrix()
            diag_K_e = np.diag(K_e)  # Extrai apenas a diagonal da matriz local

            # Mapeia a diagonal local para os índices globais
            for local_idx, global_idx in enumerate(el.dofs):
                diag_K[global_idx] += diag_K_e[local_idx]

        # Identifica os GDLs com rigidez efetivamente nula
        tol = 1e-10  # Tolerância para erros de arredondamento de ponto flutuante

        for dof in range(self.total_dofs):
            if abs(diag_K[dof]) < tol:
                if dof not in self.boundarycond:
                    self.boundarycond.append(dof)

        # Atualiza o array de GDLs livres para o solver
        self._update_free_dofs()

    def apply_boundary_condition(
        self, node_id: int, direction: str | list[str]
    ) -> None:
        """
        Restringe o deslocamento de um nó em uma ou mais direções.
        direction: 'X', 'Y', 'Z', 'RX', 'RY', 'RZ' ou uma lista com eles.
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
                dof = no.dof_ux
            elif d == "Y":
                dof = no.dof_uy
            elif d == "Z":
                dof = no.dof_uz
            elif d == "RX":
                dof = no.dof_rx
            elif d == "RY":
                dof = no.dof_ry
            elif d == "RZ":
                dof = no.dof_rz
            else:
                raise ValueError(
                    f"Direção inválida: {d}. Deve ser X, Y, Z, RX, RY ou RZ."
                )

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
            dof = no.dof_ux
        elif direction.upper() == "Y":
            dof = no.dof_uy
        elif direction.upper() == "Z":
            dof = no.dof_uz
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
        self._auto_constrain_inactive_dofs()

        K = self.assemble_stiffness()
        M = self.assemble_mass()

        # 1. Identificar os graus de liberdade livres corretamente como um array de índices
        num_modes = min(num_modes, len(self.free_dofs_array) - 1)

        # 2. Resolve o problema de autovalor usando matrizes ESPARSAS
        eigenvalues_free, eigenvectors_free = eigsh(K, k=num_modes, M=M, sigma=0.1)

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
        self._auto_constrain_inactive_dofs()

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
        respostas = np.zeros((self.total_dofs, len(freq_range_hz)), dtype=np.complex128)

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

    def map_results_to_nodes(self, global_vector: npt.NDArray[np.float64]) -> None:
        """
        Lê um vetor de resultados globais (ex: autovetor de um modo de vibrar)
        e distribui os valores para os atributos de resultado físico de cada nó.
        """
        for no in self.nodes:
            no.res_ux = global_vector[no.dof_ux]
            no.res_uy = global_vector[no.dof_uy]
            no.res_uz = global_vector[no.dof_uz]
            no.res_rx = global_vector[no.dof_rx]
            no.res_ry = global_vector[no.dof_ry]
            no.res_rz = global_vector[no.dof_rz]
