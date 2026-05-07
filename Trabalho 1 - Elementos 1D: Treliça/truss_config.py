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
class sistema:
    nodes: dict[
        int, tuple[float, float, float]
    ]  # dicionário de nós com id e tupla tupla de coordenadas (x,y,z)
    elem: dict[
        int, tuple[int, int]
    ]  # dicionário de elementos contendo id e uma tupla (nó_i, nó_j)
    material: material


NODES: dict[int, tuple[float, float, float]] = {  # em metros
    # Banzo Inferior
    0: (0.0, 0.0, 0.0),  # Apoio Esquerdo
    1: (2.0, 0.0, 0.0),
    2: (4.0, 0.0, 0.0),
    3: (6.0, 0.0, 0.0),
    4: (8.0, 0.0, 0.0),
    5: (10.0, 0.0, 0.0),  # Apoio Direito
    # Banzo Superior (deslocado 1m para formar os triângulos)
    6: (1.0, 2.0, 0.0),
    7: (3.0, 2.0, 0.0),
    8: (5.0, 2.0, 0.0),
    9: (7.0, 2.0, 0.0),
    10: (9.0, 2.0, 0.0),
}

ELEMENTS: dict[int, tuple[int, int]] = {
    # Banzo Inferior
    0: (0, 1),
    1: (1, 2),
    2: (2, 3),
    3: (3, 4),
    4: (4, 5),
    # Banzo Superior
    5: (6, 7),
    6: (7, 8),
    7: (8, 9),
    8: (9, 10),
    # Diagonais
    9: (0, 6),
    10: (6, 1),
    11: (1, 7),
    12: (7, 2),
    13: (2, 8),
    14: (8, 3),
    15: (3, 9),
    16: (9, 4),
    17: (4, 10),
    18: (10, 5),
}

aco = material(modElast=210e9, coefPoiss=0.3, dens=7850, A=0.005, name="ACO")


trelica: sistema = sistema(nodes=NODES, elem=ELEMENTS, material=aco)
