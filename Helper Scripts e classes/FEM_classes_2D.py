from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override

import numpy as np
import numpy.typing as npt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh, spsolve


@dataclass
class Node2D:
    """
    Representa um nó para problemas de Estado Plano (2 GDLs).
    """

    id: int
    x: float
    y: float

    # Índices dos 2 GDLs (Apenas translações no plano XY)
    dof_ux: int = field(init=False)
    dof_uy: int = field(init=False)

    # Resultados Físicos armazenados no próprio nó
    res_ux: float = 0.0
    res_uy: float = 0.0

    def __post_init__(self):
        self.dof_ux = 2 * self.id
        self.dof_uy = 2 * self.id + 1


class Element2D(ABC):
    """
    Classe base abstrata para elementos de área em Estado Plano.
    """

    def __init__(
        self,
        id: int,
        list_nodes: list[Node2D],
        E: float,
        nu: float,
        thickness: float,
        rho: float,
    ):
        self.id = id
        self.nodes = list_nodes
        self.E = E
        self.nu = nu
        self.thickness = thickness
        self.rho = rho  # Nova propriedade!

        self.dofs = []
        for no in self.nodes:
            self.dofs.extend([no.dof_ux, no.dof_uy])

    def get_constitutive_matrix(self) -> npt.NDArray[np.float64]:
        """
        Retorna a matriz de elasticidade [D] para Estado Plano de Tensão (Plane Stress).
        Assume material isotrópico.
        """
        coef = self.E / (1.0 - self.nu**2)
        return coef * np.array(
            [
                [1.0, self.nu, 0.0],
                [self.nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - self.nu) / 2.0],
            ]
        )

    @abstractmethod
    def get_stiffness_matrix(self) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_mass_matrix(self) -> npt.NDArray[np.float64]:
        pass


class TriangleCST(Element2D):
    """
    Elemento Triangular de Deformação Constante (CST - 3 Nós).
    """

    def __init__(
        self,
        id: int,
        list_nodes: list[Node2D],
        E: float,
        nu: float,
        thickness: float,
        rho: float,
    ):
        if len(list_nodes) != 3:
            raise ValueError("O elemento CST exige exatamente 3 nós.")
        super().__init__(id, list_nodes, E, nu, thickness, rho)

    @override
    def get_stiffness_matrix(self) -> npt.NDArray[np.float64]:
        # Extrai coordenadas para facilitar a leitura
        x1, y1 = self.nodes[0].x, self.nodes[0].y
        x2, y2 = self.nodes[1].x, self.nodes[1].y
        x3, y3 = self.nodes[2].x, self.nodes[2].y

        # Calcula a área do triângulo usando determinante
        det_J = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        A = abs(det_J) / 2.0

        if A <= 1e-12:
            raise ValueError(f"Elemento {self.id} possui área nula ou nós colineares.")

        # Constantes geométricas (diferenças de coordenadas)
        b1, c1 = y2 - y3, x3 - x2
        b2, c2 = y3 - y1, x1 - x3
        b3, c3 = y1 - y2, x2 - x1

        # Matriz cinemática [B] (Deformação-Deslocamento) de tamanho 3x6
        B = (1.0 / (2.0 * A)) * np.array(
            [
                [b1, 0.0, b2, 0.0, b3, 0.0],
                [0.0, c1, 0.0, c2, 0.0, c3],
                [c1, b1, c2, b2, c3, b3],
            ]
        )

        # Matriz constitutiva [D]
        D = self.get_constitutive_matrix()

        # Matriz de Rigidez: [K_e] = [B]^T * [D] * [B] * Area * espessura
        K_e = B.T @ D @ B * A * self.thickness
        return K_e

    @override
    def get_mass_matrix(self) -> npt.NDArray[np.float64]:
        # Recalcula a área rapidamente
        x1, y1 = self.nodes[0].x, self.nodes[0].y
        x2, y2 = self.nodes[1].x, self.nodes[1].y
        x3, y3 = self.nodes[2].x, self.nodes[2].y
        A = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0

        # Coeficiente da matriz de massa da apostila
        coef = (self.rho * self.thickness * A) / 12.0

        # Matriz 6x6 consistente (Eisenberg & Malvern)
        m_e = coef * np.array(
            [
                [2.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 2.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 2.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 2.0],
            ]
        )
        return m_e


class FEMSystem2D:
    """
    Solver para problemas de Estado Plano (2 GDLs por nó).
    """

    def __init__(self, nodes: list[Node2D], elements: list[Element2D]):
        self.nodes = nodes
        self.elements = elements
        self.boundarycond: list[int] = []
        self.total_dofs: int = len(nodes) * 2  # Apenas 2 GDLs no 2D!
        self.nodal_loads: dict[int, float] = {}
        self._update_free_dofs()

    def _update_free_dofs(self) -> None:
        all_dofs = np.arange(self.total_dofs, dtype="int32")
        unique_bc = sorted(list(set(self.boundarycond)))
        self.free_dofs_array = np.delete(all_dofs, unique_bc)

    def apply_boundary_condition(
        self, node_id: int, direction: str | list[str]
    ) -> None:
        no = next((n for n in self.nodes if n.id == node_id), None)
        if no is None:
            raise ValueError(f"Nó {node_id} não encontrado.")

        directions = (
            [direction.upper()]
            if isinstance(direction, str)
            else [d.upper() for d in direction]
        )

        for d in directions:
            if d == "X":
                dof = no.dof_ux
            elif d == "Y":
                dof = no.dof_uy
            else:
                raise ValueError(f"Direção {d} inválida em 2D. Use X ou Y.")

            if dof not in self.boundarycond:
                self.boundarycond.append(dof)
        self._update_free_dofs()

    def apply_load(self, node_id: int, direction: str, magnitude: float) -> None:
        no = next((n for n in self.nodes if n.id == node_id), None)
        if no is None:
            raise ValueError(f"Nó {node_id} não encontrado.")

        if direction.upper() == "X":
            dof = no.dof_ux
        elif direction.upper() == "Y":
            dof = no.dof_uy
        else:
            raise ValueError("Direção em 2D deve ser X ou Y.")

        self.nodal_loads[dof] = self.nodal_loads.get(dof, 0.0) + magnitude

    def assemble_stiffness(self):
        K_global = lil_matrix((self.total_dofs, self.total_dofs), dtype=np.float64)
        for el in self.elements:
            K_e = el.get_stiffness_matrix()
            grid = np.ix_(el.dofs, el.dofs)
            K_global[grid] += K_e
        return K_global.tocsr()

    def solve_static(self) -> npt.NDArray[np.float64]:
        """Resolve F = K*U para casos estáticos (sem inércia)."""
        K = self.assemble_stiffness()
        F_global = np.zeros(self.total_dofs, dtype=np.float64)
        for dof, mag in self.nodal_loads.items():
            F_global[dof] = mag

        K_free = K[np.ix_(self.free_dofs_array, self.free_dofs_array)]
        F_free = F_global[self.free_dofs_array]

        u_free = spsolve(K_free, F_free)

        U_full = np.zeros(self.total_dofs, dtype=np.float64)
        U_full[self.free_dofs_array] = u_free
        return U_full

    def map_results_to_nodes(self, global_vector: npt.NDArray[np.float64]) -> None:
        """Injeta as respostas de volta para a visualização limpa."""
        for no in self.nodes:
            no.res_ux = global_vector[no.dof_ux]
            no.res_uy = global_vector[no.dof_uy]

    def assemble_mass(self):
        M_global = lil_matrix((self.total_dofs, self.total_dofs), dtype=np.float64)
        for el in self.elements:
            M_e = el.get_mass_matrix()
            grid = np.ix_(el.dofs, el.dofs)
            M_global[grid] += M_e
        return M_global.tocsr()

    def solve_modal_analysis(
        self, num_modes=6
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Calcula frequências naturais (Hz) e formas modais.
        """
        K = self.assemble_stiffness()
        M = self.assemble_mass()

        K_free = K[np.ix_(self.free_dofs_array, self.free_dofs_array)]
        M_free = M[np.ix_(self.free_dofs_array, self.free_dofs_array)]

        num_modes = min(num_modes, len(self.free_dofs_array) - 1)

        # Extração de autovalores
        eigenvalues_free, eigenvectors_free = eigsh(
            K_free, k=num_modes, M=M_free, sigma=0.1
        )

        # Ordenação
        idx = eigenvalues_free.argsort()
        eigenvalues_free = eigenvalues_free[idx]
        eigenvectors_free = eigenvectors_free[:, idx]

        # Conversão para Hz
        natural_frequencies = np.sqrt(np.abs(eigenvalues_free)) / (2 * np.pi)

        # Remontagem para o tamanho total do sistema
        eigenvectors = np.zeros((self.total_dofs, num_modes))
        eigenvectors[self.free_dofs_array, :] = eigenvectors_free

        return natural_frequencies, eigenvectors
