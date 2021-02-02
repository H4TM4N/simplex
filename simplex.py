import numpy as np

# 目的関数用クラス
class S_z:
    val: float  # 値
    data: np.ndarray  # 係数

    def __init__(self, val, data):
        self.val = val
        self.data = data

# 制約条件(基底形式)用クラス
class S_line(S_z):
    name: str  # 基底変数名
    val: float  # その値
    data: np.ndarray  # 係数

    def __init__(self, name, val, data):
        self.name = name
        self.val = val
        self.data = data

# pivot列の選択(もっとも大きい正の係数を持つ列)
def search_pivot_column(z: S_z) -> int:
    idx = np.argmax(z.data)
    if z.data[idx] <= 0:
        return -1
    return idx

# pivot行の選択(比の計算が最小となる行)
def search_pivot_row(ls: list, pc: int) -> int:
    vs = []
    for l in ls:
        tmp = l.val/l.data[pc]
        if tmp < 0:
            vs = np.append(vs, float('inf'))
        else:
            vs = np.append(vs, l.val/l.data[pc])
    return np.argmin(vs)

# pivot要素を1に変換
def divline(l: S_line, p_val: float):
    l.val = l.val / p_val
    l.data = l.data / p_val

# pivot演算
def minusline(l: S_z, pl: S_line, pc: int):
    mag = l.data[pc]
    l.val = l.val - pl.val*mag
    l.data = l.data - pl.data*mag

# 小数点以下n桁に四捨五入
def round_line(data: list, n: int) -> list:
    return list(map(lambda x: str(round(x, n)), data))

# pivot要素を反転表示
def accent(data: list, pc: int) -> list:
    if pc == -1:
        return "\t".join(data)
    data[pc] = '\033[07m' + data[pc] + '\033[0m'
    return "\t".join(data)

# シンプレックス表の表示
def print_simplex(z: S_z, ls: S_line, n: int, pc: int = -1, pr: int = -1):
    print(f" z: {round(z.val, n)}  \t{accent(round_line(z.data, n), -1)}")
    for r, l in enumerate(ls):
        if r == pr:
            print(f"{l.name}: {round(l.val, n)}  \t{accent(round_line(l.data, n), pc)}")
        else:
            print(f"{l.name}: {round(l.val, n)}  \t{accent(round_line(l.data, n), -1)}")

# ----------------------------------------------------------
# DATA EXAMPLE
# ----------------------------------------------------------
# min. w = y1 + y2 + y3
# min. z = 3x1 + 2x2
# s.t. 2x1 + x2 -x3 + y1 = 20
#      4x1 + 3x2 -x4 + y2 = 56
#      5x1 + 4x2 -x5 + y3 = 73
#      x1, x2, x3, x4, x5, y1, y2, y3 >= 0
# ----------------------------------------------------------
# Penalty Method(罰金法)
# ----------------------------------------------------------
# min. z = r(y1 + y2 + y3) + 3x1 + 2x2
#        = 149r + (-11r+3)x1 + (-8r+2)x2 + rx3 + rx4 + rx5
#        <-> z + (11r-3)x1 + (8r-2)x2 - rx3 - rx4 -rx5 = 149r
# s.t. 2x1 + x2 -x3 + y1 = 20
#      4x1 + 3x2 -x4 + y2 = 56
#      5x1 + 4x2 -x5 + y3 = 73
#      x1, x2, x3, x4, x5, y1, y2, y3 >= 0
# ----------------------------------------------------------
# simplex table
# ----------------------------------------------------------
#    |   v  |  x1  |  x2  |  x3  |  x4  |  x5  |  y1  |  y2  |  y3  |
#  z | 149r | 11r-3| 8r-2 |  -r  |  -r  |  -r  |   0  |   0  |   0  |
# y1 |  20  |   2  |   1  |  -1  |   0  |   0  |   1  |   0  |   0  |
# y2 |  56  |   4  |   3  |   0  |  -1  |   0  |   0  |   1  |   0  |
# y3 |  73  |   5  |   4  |   0  |   0  |  -1  |   0  |   0  |   1  |
# ----------------------------------------------------------

r = 100.0  # 罰金法における充分大きな係数r
names = ['x1', 'x2', 'x3', 'x4', 'x5', 'y1', 'y2', 'y3']  # 変数名
# simplex表の値
z = S_z(149.0*r, np.array([11.0*r-3.0, 8.0*r-2.0, -r, -r, -r, 0.0, 0.0, 0.0]))
l1 = S_line('y1', 20.0, np.array([2.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
l2 = S_line('y2', 56.0, np.array([4.0, 3.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0]))
l3 = S_line('y3', 73.0, np.array([5.0, 4.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0]))


ls = [l1, l2, l3]  # 制約条件式クラス管理用
# print_simplex(z, ls, 2)
while True:
    if (pc := search_pivot_column(z)) == -1:
        break
    pr = search_pivot_row(ls, pc)
    print_simplex(z, ls, 2, pc=pc, pr=pr)
    divline(ls[pr], ls[pr].data[pc])
    # print(f"PivPos -- column:{names[pc]}, row:{ls[pr].name}")
    ls[pr].name = names[pc]
    for i, l in enumerate(ls):
        if i == pr:
            continue
        minusline(l, ls[pr], pc)
    minusline(z, ls[pr], pc)
    print('---------------------------------------')

print_simplex(z, ls, 2)

