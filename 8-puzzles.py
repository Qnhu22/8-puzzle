import tkinter as tk
from tkinter import messagebox, ttk
from collections import deque
import heapq
import time
import random
import math
import sys

# Tăng giới hạn đệ quy (dự phòng, không khuyến khích sử dụng làm giải pháp chính)
sys.setrecursionlimit(2000)

# ===============================
# LỚP GIẢI BÀI TOÁN 8-PUZZLE
# ===============================
class EightPuzzle:
    def __init__(self, initial_state, goal_state):
        """Khởi tạo bài toán 8-puzzle với trạng thái ban đầu và mục tiêu."""
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Di chuyển: lên, xuống, trái, phải

    def find_blank(self, state):
        """Tìm vị trí ô trống (giá trị 0) trong trạng thái."""
        for row in range(3):
            for col in range(3):
                if state[row][col] == 0:
                    return row, col
        raise ValueError("Trạng thái không hợp lệ: Không tìm thấy ô trống.")

    def swap_tiles(self, state, r1, c1, r2, c2):
        """Hoán đổi hai ô trong trạng thái."""
        new_state = [row[:] for row in state]
        new_state[r1][c1], new_state[r2][c2] = new_state[r2][c2], new_state[r1][c1]
        return tuple(map(tuple, new_state))

    def get_neighbors(self, state):
        """Tạo các trạng thái lân cận bằng cách di chuyển ô trống."""
        empty_row, empty_col = self.find_blank(state)
        neighbors = []
        for move_row, move_col in self.moves:
            new_row, new_col = empty_row + move_row, empty_col + move_col
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = [list(row) for row in state]
                new_state[empty_row][empty_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[empty_row][empty_col]
                neighbors.append(tuple(map(tuple, new_state)))
        return neighbors

    def heuristic(self, state):
        """Tính heuristic: khoảng cách Manhattan."""
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    goal_x, goal_y = divmod(state[i][j] - 1, 3)
                    distance += abs(i - goal_x) + abs(j - goal_y)
        return distance

    def is_solvable(self):
        """Kiểm tra xem bài toán có thể giải được hay không."""
        def get_inversions(state):
            flat = sum(state, [])
            inversions = 0
            for i in range(len(flat)):
                if flat[i] == 0:
                    continue
                for j in range(i + 1, len(flat)):
                    if flat[j] == 0:
                        continue
                    if flat[i] > flat[j]:
                        inversions += 1
            return inversions
        initial_inversions = get_inversions(self.initial_state)
        goal_inversions = get_inversions(self.goal_state)
        return (initial_inversions % 2) == (goal_inversions % 2)

    # ===============================
    # THUẬT TOÁN TÌM KIẾM
    # ===============================

    def solve_bfs(self):
        """Tìm kiếm theo chiều rộng (Breadth-First Search) - Thực hiện duyệt theo mức."""
        start_time = time.time()
        queue = deque([(self.initial_state, [])])
        visited = {tuple(map(tuple, self.initial_state))}
        count = 1
        while queue:
            current, path = queue.popleft()
            if current == self.goal_state:
                return path + [current], time.time() - start_time, count
            for neighbor in self.get_neighbors(current):
                t = neighbor
                if t not in visited:
                    visited.add(t)
                    queue.append((neighbor, path + [neighbor]))
                    count += 1
        return None, time.time() - start_time, count

    def solve_dfs(self, max_depth=100, max_states=200000):
        """Tìm kiếm theo chiều sâu (Depth-First Search) - Duyệt sâu nhất trước."""
        if not self.is_solvable():
            return None, 0, 0
        start_time = time.time()
        stack = [(self.initial_state, 0, [])]
        visited = {tuple(map(tuple, self.initial_state))}
        count = 1
        while stack and count < max_states:
            current, depth, path = stack.pop()
            if current == self.goal_state:
                return path + [current], time.time() - start_time, count
            if depth < max_depth:
                for neighbor in self.get_neighbors(current):
                    t = neighbor
                    if t not in visited:
                        visited.add(t)
                        stack.append((neighbor, depth + 1, path + [neighbor]))
                        count += 1
        return None, time.time() - start_time, count

    def solve_ids(self):
        """Tìm kiếm lặp sâu dần (Iterative Deepening Search) - Kết hợp DFS và giới hạn độ sâu."""
        start_time = time.time()
        depth = 0
        while True:
            visited = set()
            counter = [1]
            result = self.depth_limited_search(self.initial_state, [], depth, visited, counter)
            if result:
                return result, time.time() - start_time, counter[0]
            depth += 1
            if depth > 100:
                return None, time.time() - start_time, counter[0]

    def depth_limited_search(self, state, path, depth, visited, counter):
        """Hỗ trợ IDS - Tìm kiếm với giới hạn độ sâu."""
        if state == self.goal_state:
            return path + [state]
        if depth == 0:
            return None
        visited.add(tuple(map(tuple, state)))
        for neighbor in self.get_neighbors(state):
            t = neighbor
            if t not in visited:
                counter[0] += 1
                result = self.depth_limited_search(neighbor, path + [neighbor], depth - 1, visited, counter)
                if result:
                    return result
        return None

    def solve_ucs(self):
        """Tìm kiếm chi phí đồng nhất (Uniform Cost Search) - Ưu tiên chi phí thấp nhất."""
        start_time = time.time()
        pq = [(0, self.initial_state, [])]
        visited = {tuple(map(tuple, self.initial_state))}
        count = 1
        while pq:
            cost, current, path = heapq.heappop(pq)
            if current == self.goal_state:
                return path + [current], time.time() - start_time, count
            for neighbor in self.get_neighbors(current):
                t = neighbor
                if t not in visited:
                    heapq.heappush(pq, (cost + 1, neighbor, path + [neighbor]))
                    visited.add(t)
                    count += 1
        return None, time.time() - start_time, count

    def solve_greedy(self):
        """Tìm kiếm tham lam (Greedy Search) - Sử dụng heuristic để chọn bước tiếp theo."""
        start_time = time.time()
        pq = [(self.heuristic(self.initial_state), self.initial_state, [])]
        visited = {tuple(map(tuple, self.initial_state))}
        count = 1
        while pq:
            _, current, path = heapq.heappop(pq)
            if current == self.goal_state:
                return path + [current], time.time() - start_time, count
            for neighbor in self.get_neighbors(current):
                t = neighbor
                if t not in visited:
                    heapq.heappush(pq, (self.heuristic(neighbor), neighbor, path + [neighbor]))
                    visited.add(t)
                    count += 1
        return None, time.time() - start_time, count

    def solve_astar(self):
        """Tìm kiếm A* - Kết hợp chi phí và heuristic."""
        start_time = time.time()
        pq = [(self.heuristic(self.initial_state), 0, self.initial_state, [])]
        visited = {}
        count = 1
        while pq:
            f, g, current, path = heapq.heappop(pq)
            if current == self.goal_state:
                return path + [current], time.time() - start_time, count
            visited[tuple(map(tuple, current))] = g
            for neighbor in self.get_neighbors(current):
                new_g = g + 1
                new_f = new_g + self.heuristic(neighbor)
                t = neighbor
                if t not in visited or new_g < visited[t]:
                    heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
                    visited[t] = new_g
                    count += 1
        return None, time.time() - start_time, count

    def solve_idastar(self):
        """Tìm kiếm A* lặp - Kết hợp A* với giới hạn chi phí."""
        start_time = time.time()
        threshold = self.heuristic(self.initial_state)
        count = [1]
        def search(path, g):
            current = path[-1]
            f = g + self.heuristic(current)
            if f > threshold:
                return f
            if current == self.goal_state:
                return path
            min_threshold = float('inf')
            for neighbor in self.get_neighbors(current):
                if neighbor not in path:
                    count[0] += 1
                    result = search(path + [neighbor], g + 1)
                    if isinstance(result, list):
                        return result
                    if result < min_threshold:
                        min_threshold = result
            return min_threshold
        while True:
            result = search([self.initial_state], 0)
            if isinstance(result, list):
                return result, time.time() - start_time, count[0]
            if result == float('inf'):
                return None, time.time() - start_time, count[0]
            threshold = result

    def solve_simple_hill_climbing(self):
        """Leo đồi đơn giản - Chọn bước cải thiện đầu tiên."""
        start_time = time.time()
        current = self.initial_state
        path = [current]
        count = 1
        while True:
            for neighbor in self.get_neighbors(current):
                count += 1
                if self.heuristic(neighbor) < self.heuristic(current):
                    current = neighbor
                    path.append(current)
                    break
            else:
                break
        return path, time.time() - start_time, count

    def solve_steepest_ascent_hill_climbing(self):
        """Leo đồi dốc nhất - Chọn bước cải thiện tốt nhất."""
        start_time = time.time()
        current = self.initial_state
        path = [current]
        count = 1
        while True:
            neighbors = self.get_neighbors(current)
            count += len(neighbors)
            best = min(neighbors, key=self.heuristic, default=None)
            if best and self.heuristic(best) < self.heuristic(current):
                current = best
                path.append(current)
            else:
                break
        return path, time.time() - start_time, count

    def solve_stochastic_hill_climbing(self):
        """Leo đồi ngẫu nhiên - Chọn ngẫu nhiên bước cải thiện."""
        start_time = time.time()
        current = self.initial_state
        path = [current]
        count = 1
        while True:
            neighbors = self.get_neighbors(current)
            better = [n for n in neighbors if self.heuristic(n) < self.heuristic(current)]
            count += len(neighbors)
            if better:
                current = random.choice(better)
                path.append(current)
            else:
                break
        return path, time.time() - start_time, count

    def solve_simulated_annealing(self):
        """Mô phỏng luyện kim - Cho phép chấp nhận bước xấu với xác suất."""
        start_time = time.time()
        current = self.initial_state
        path = [current]
        count = 1
        T = 100.0
        T_min = 0.01
        alpha = 0.99
        while T > T_min:
            neighbors = self.get_neighbors(current)
            if not neighbors:
                break
            next_state = random.choice(neighbors)
            delta = self.heuristic(next_state) - self.heuristic(current)
            count += 1
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = next_state
                path.append(current)
                if current == self.goal_state:
                    return path, time.time() - start_time, count
            T *= alpha
        return path, time.time() - start_time, count

    def solve_beam_search(self, beam_width=10, max_iterations=1000):
        """Tìm kiếm chùm - Giới hạn số lượng đường dẫn tốt nhất."""
        start_time = time.time()
        beam = [(self.heuristic(self.initial_state), self.initial_state, [self.initial_state])]
        visited = {tuple(map(tuple, self.initial_state))}
        count = 1
        iterations = 0
        while beam and iterations < max_iterations:
            iterations += 1
            next_beam = []
            for _, current_state, path in beam:
                if current_state == self.goal_state:
                    return path, time.time() - start_time, count
                neighbors = self.get_neighbors(current_state)
                count += len(neighbors)
                for neighbor in neighbors:
                    neighbor_tuple = neighbor
                    if neighbor_tuple not in visited:
                        visited.add(neighbor_tuple)
                        new_path = path + [neighbor]
                        heapq.heappush(next_beam, (self.heuristic(neighbor), neighbor, new_path))
            beam = heapq.nsmallest(beam_width, next_beam, key=lambda item: item[0])
        return None, time.time() - start_time, count

    def solve_genetic_algorithm(self, population_size=50, generations=500, mutation_rate=0.05):
        """Thuật toán di truyền - Tiến hóa quần thể để tìm lời giải."""
        if not self.is_solvable():
            return None, 0, 0
        def mutate(state):
            neighbors = self.get_neighbors(state)
            return random.choice(neighbors) if neighbors else state
        def crossover(p1, p2):
            flat1 = sum(p1, [])
            flat2 = sum(p2, [])
            cut = random.randint(1, 8)
            child = flat1[:cut] + flat2[cut:]
            count = {}
            for num in child:
                count[num] = count.get(num, 0) + 1
            missing = set(range(9)) - set(count.keys())
            excess = []
            for num, freq in count.items():
                if freq > 1:
                    excess.extend([num] * (freq - 1))
            child_corrected = child[:]
            for i in range(len(child_corrected)):
                if count[child_corrected[i]] > 1 and missing:
                    child_corrected[i] = missing.pop()
                    count[child_corrected[i]] = 1
            return [child_corrected[i:i + 3] for i in range(0, 9, 3)]
        def fitness(state):
            return -self.heuristic(state)
        start_time = time.time()
        population = [self.initial_state]
        for _ in range(population_size - 1):
            neighbors = self.get_neighbors(random.choice(population))
            population.append(random.choice(neighbors) if neighbors else self.initial_state)
        count = len(population)
        for _ in range(generations):
            population.sort(key=fitness, reverse=True)
            next_gen = population[:2]
            while len(next_gen) < population_size:
                parents = random.sample(population[:5], 2)
                child = crossover(parents[0], parents[1])
                if random.random() < mutation_rate:
                    child = mutate(child)
                next_gen.append(child)
                count += 1
            population = next_gen
            current_best = max(population, key=fitness)
            if self.heuristic(current_best) == 0:
                return [self.initial_state, current_best], time.time() - start_time, count
        best = max(population, key=fitness)
        return [self.initial_state, best], time.time() - start_time, count

    def and_or_search(self, max_steps=1000):
        """AND-OR Search với trạng thái khởi đầu cụ thể."""
        start_time = time.time()
        initial_state = tuple(map(tuple, self.initial_state))
        queue = deque([(initial_state, [], {initial_state})])  # (state, path, belief_set)
        visited = set()
        explored_states = set()
        count = 0

        while queue and count < max_steps:
            state, path, belief_set = queue.popleft()
            count += 1
            explored_states.add(state)

            # Kiểm tra nếu tất cả trạng thái trong belief_set đều đạt mục tiêu
            if all(s == tuple(map(tuple, self.goal_state)) for s in belief_set):
                return [list(map(list, p)) for p in path] + [list(map(list, state))], time.time() - start_time, count

            belief_key = frozenset(belief_set)
            if belief_key in visited:
                continue
            visited.add(belief_key)

            # Tạo các trạng thái lân cận với sự không chắc chắn
            neighbors = self.get_neighbors(state)
            for neighbor in neighbors:
                # Mô phỏng sự không chắc chắn: có xác suất thêm các trạng thái khác
                new_belief_set = {neighbor}
                if random.random() < 0.3:  # Xác suất 30% tạo thêm trạng thái không chắc chắn
                    i, j = self.find_blank(neighbor)
                    for di, dj in self.moves:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 3 and 0 <= nj < 3:
                            uncertain_state = [list(row) for row in neighbor]
                            uncertain_state[i][j], uncertain_state[ni][nj] = uncertain_state[ni][nj], uncertain_state[i][j]
                            new_belief_set.add(tuple(map(tuple, uncertain_state)))

                # Thêm trạng thái mới vào hàng đợi
                queue.append((neighbor, path + [state], new_belief_set))

        return None, time.time() - start_time, count

    def belief_state_search(self, initial_belief=None, max_steps=1000):
        start_time = time.time()
        initial_state = tuple(map(tuple, self.initial_state))
        if initial_belief is None:
            # Tạo tập belief states: trạng thái ban đầu và tối đa 2 trạng thái lân cận
            neighbors = self.get_neighbors(initial_state)
            initial_belief = {initial_state}
            initial_belief.update(neighbors[:2])
            # Đảm bảo có ít nhất 3 trạng thái bằng cách thêm ngẫu nhiên nếu thiếu
            while len(initial_belief) < 3:
                new_state = [list(row) for row in initial_state]
                i, j = self.find_blank(initial_state)
                moves = [(di, dj) for di, dj in self.moves if 0 <= i + di < 3 and 0 <= j + dj < 3]
                if moves:
                    di, dj = random.choice(moves)
                    new_state[i][j], new_state[i + di][j + dj] = new_state[i + di][j + dj], new_state[i][j]
                    initial_belief.add(tuple(map(tuple, new_state)))

        queue = deque([(initial_belief, [])])
        visited = set()
        explored = set(initial_belief)
        count = 0

        while queue and count < max_steps:
            belief, path = queue.popleft()
            count += 1
            belief_key = frozenset(belief)
            if belief_key in visited:
                continue
            visited.add(belief_key)

            if all(b == tuple(map(tuple, self.goal_state)) for b in belief):
                return path + [list(map(list, min(belief, key=self.heuristic)))], time.time() - start_time, count, belief

            for action in range(4):
                new_belief = set()
                for state in belief:
                    neighbors = self.get_neighbors(state)
                    if action < len(neighbors):
                        next_state = neighbors[action]
                        new_belief.add(next_state)
                        if random.random() < 0.1:
                            i, j = self.find_blank(next_state)
                            for di, dj in self.moves:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < 3 and 0 <= nj < 3:
                                    uncertain_state = [list(row) for row in next_state]
                                    uncertain_state[i][j], uncertain_state[ni][nj] = uncertain_state[ni][nj], uncertain_state[i][j]
                                    new_belief.add(tuple(map(tuple, uncertain_state)))

                if new_belief:
                    best_states = sorted(new_belief, key=self.heuristic)[:3]
                    new_belief_set = set(best_states)
                    queue.append((new_belief_set, path + [list(map(list, min(belief, key=self.heuristic)))]))
                    explored.update(new_belief_set)

        return None, time.time() - start_time, count, initial_belief

    def partial_observable_search(self, max_steps=1000):
        start_time = time.time()
        def one_at_00(state):
            return state[0][0] == 1

        initial_state = tuple(map(tuple, self.initial_state))
        belief = {initial_state} if one_at_00(initial_state) else set()
        for neighbor in self.get_neighbors(initial_state):
            if one_at_00(neighbor):
                belief.add(neighbor)

        if not belief:
            return None, time.time() - start_time, 0, set()

        belief = sorted(belief, key=self.heuristic)[:3]
        queue = deque([(set(belief), [list(map(list, initial_state))])])
        visited = set()
        count = 0

        while queue and count < max_steps:
            belief_state, path = queue.popleft()
            count += 1
            belief_key = frozenset(belief_state)
            if belief_key in visited:
                continue
            visited.add(belief_key)

            if all(b == tuple(map(tuple, self.goal_state)) for b in belief_state):
                return path + [list(map(list, self.goal_state))], time.time() - start_time, count, belief_state

            for action in range(4):
                new_belief = set()
                for state in belief_state:
                    neighbors = self.get_neighbors(state)
                    if action < len(neighbors):
                        next_state = neighbors[action]
                        if one_at_00(next_state):
                            new_belief.add(next_state)
                        if random.random() < 0.1:
                            i, j = self.find_blank(next_state)
                            for di, dj in self.moves:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < 3 and 0 <= nj < 3:
                                    uncertain_state = [list(row) for row in next_state]
                                    uncertain_state[i][j], uncertain_state[ni][nj] = uncertain_state[ni][nj], uncertain_state[i][j]
                                    if one_at_00(tuple(map(tuple, uncertain_state))):
                                        new_belief.add(tuple(map(tuple, uncertain_state)))

                if new_belief:
                    top_3 = sorted(new_belief, key=self.heuristic)[:3]
                    new_belief_set = set(top_3)
                    representative_state = list(map(list, min(belief_state, key=self.heuristic)))
                    queue.append((new_belief_set, path + [representative_state]))

        return None, time.time() - start_time, count, belief

    def solve_belief_state_and_or_search(self, max_iterations=10000):
        return self.belief_state_search(max_steps=max_iterations)

    def solve_partial_observable_search(self):
        return self.partial_observable_search()

    def solve_backtracking(self, max_depth=100):
        """Tìm kiếm quay lui - Gán giá trị từ trạng thái rỗng đến trạng thái mục tiêu."""
        start_time = time.time()
        state = [[-1 for _ in range(3)] for _ in range(3)]
        count = [1]

        def backtrack(state, pos, used):
            if pos == 9:
                if state == self.goal_state:
                    return [state]
                return None
            row, col = divmod(pos, 3)
            for val in range(9):
                if val not in used:
                    state[row][col] = val
                    used.add(val)
                    count[0] += 1
                    result = backtrack(state, pos + 1, used)
                    if result:
                        return result
                    state[row][col] = -1
                    used.remove(val)
            return None

        used = set()
        result = backtrack(state, 0, used)
        if result:
            return [result], time.time() - start_time, count[0]
        return None, time.time() - start_time, count[0]

    def solve_forward_checking(self):
        """Tìm kiếm với kiểm tra tiến - Gán giá trị từ trạng thái rỗng, thu hẹp domain."""
        start_time = time.time()
        state = [[-1 for _ in range(3)] for _ in range(3)]
        count = [1]

        domains = {(i, j): list(range(9)) for i in range(3) for j in range(3)}

        def forward_check(state, pos, domains):
            if pos == 9:
                if state == self.goal_state:
                    return [state]
                return None
            row, col = divmod(pos, 3)
            for val in domains[(row, col)]:
                state[row][col] = val
                count[0] += 1
                new_domains = {k: v[:] for k, v in domains.items()}
                consistent = True
                for ni in range(3):
                    for nj in range(3):
                        if state[ni][nj] == -1 and val in new_domains[(ni, nj)]:
                            new_domains[(ni, nj)].remove(val)
                            if not new_domains[(ni, nj)]:
                                consistent = False
                                break
                    if not consistent:
                        break
                if consistent:
                    result = forward_check(state, pos + 1, new_domains)
                    if result:
                        return result
                state[row][col] = -1
            return None

        result = forward_check(state, 0, domains)
        if result:
            return [result], time.time() - start_time, count[0]
        return None, time.time() - start_time, count[0]

    def solve_min_conflicts(self, max_steps=1000, max_restarts=5):
        """Tìm kiếm xung đột tối thiểu - Gán ngẫu nhiên và giảm xung đột."""
        start_time = time.time()
        count = [1]

        def initialize_state():
            state = [[-1 for _ in range(3)] for _ in range(3)]
            values = list(range(9))
            random.shuffle(values)
            idx = 0
            for i in range(3):
                for j in range(3):
                    state[i][j] = values[idx]
                    idx += 1
            return state

        def get_conflicts(state, row, col):
            val = state[row][col]
            target_val = self.goal_state[row][col]
            return 1 if val != target_val else 0

        for restart in range(max_restarts):
            state = initialize_state()
            for step in range(max_steps):
                count[0] += 1
                if state == self.goal_state:
                    return [state], time.time() - start_time, count[0]
                
                conflicts = []
                for i in range(3):
                    for j in range(3):
                        if get_conflicts(state, i, j) > 0:
                            conflicts.append((i, j))
                
                if not conflicts:
                    break
                
                row, col = random.choice(conflicts)
                
                best_val = state[row][col]
                best_conflicts = float('inf')
                for val in range(9):
                    used = False
                    for i in range(3):
                        for j in range(3):
                            if (i != row or j != col) and state[i][j] == val:
                                used = True
                                break
                        if used:
                            break
                    if used:
                        continue
                    old_val = state[row][col]
                    state[row][col] = val
                    curr_conflicts = sum(get_conflicts(state, i, j) for i in range(3) for j in range(3))
                    if curr_conflicts < best_conflicts:
                        best_conflicts = curr_conflicts
                        best_val = val
                    state[row][col] = old_val
                
                if best_val != state[row][col]:
                    for i in range(3):
                        for j in range(3):
                            if (i != row or j != col) and state[i][j] == best_val:
                                state[i][j] = state[row][col]
                                break
                        else:
                            continue
                        break
                    state[row][col] = best_val
        
        return None, time.time() - start_time, count[0]

    def solve_q_learning(self, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
        """Học tăng cường với Q-Learning - Tích lũy kinh nghiệm qua các lần thử."""
        start_time = time.time()
        q_table = {}
        def state_to_tuple(state):
            return tuple(map(tuple, state))
        def get_action(state):
            neighbors = self.get_neighbors(state)
            actions = []
            for neighbor in neighbors:
                actions.append(neighbor)
            return actions
        for _ in range(episodes):
            state = [row[:] for row in self.initial_state]
            while state != self.goal_state:
                state_tuple = state_to_tuple(state)
                if state_tuple not in q_table:
                    q_table[state_tuple] = {tuple(map(tuple, n)): 0 for n in get_action(state)}
                if random.random() < epsilon:
                    action = random.choice(list(q_table[state_tuple].keys()))
                else:
                    action = max(q_table[state_tuple], key=q_table[state_tuple].get)
                next_state = [list(row) for row in action]
                reward = -1 if next_state != self.goal_state else 100
                next_state_tuple = state_to_tuple(next_state)
                if next_state_tuple not in q_table:
                    q_table[next_state_tuple] = {tuple(map(tuple, n)): 0 for n in get_action(next_state)}
                max_future_q = max(q_table[next_state_tuple].values())
                q_table[state_tuple][action] = (1 - alpha) * q_table[state_tuple][action] + alpha * (reward + gamma * max_future_q)
                state = next_state
        path = [self.initial_state]
        state = [row[:] for row in self.initial_state]
        count = 1
        while state != self.goal_state:
            state_tuple = state_to_tuple(state)
            if state_tuple not in q_table:
                return None, time.time() - start_time, count
            action = max(q_table[state_tuple], key=q_table[state_tuple].get)
            state = [list(row) for row in action]
            path.append(state)
            count += 1
            if len(path) > 100:
                return None, time.time() - start_time, count
        return path, time.time() - start_time, count

# ===============================
# GIAO DIỆN ĐỒ HỌA TKINTER
# ===============================
class EightPuzzleGUI:
    def __init__(self, root, puzzle):
        self.root = root
        self.puzzle = puzzle
        self.original_state = [row[:] for row in puzzle.initial_state]
        self.solution_steps = None
        self.current_step = 0
        self.is_playing = False
        self.history = []
        self.performance_data = []
        self.belief_window = None

        self.root.configure(bg='#F0F8FF')
        self.custom_font = ('Helvetica', 12, 'bold')

        main_frame = tk.Frame(root, bg='#F0F8FF')
        main_frame.pack(pady=10, padx=10)

        input_frame = tk.Frame(main_frame, bg='#F0F8FF')
        input_frame.grid(row=0, column=0, columnspan=7, pady=10)
        tk.Label(input_frame, text="Trạng thái ban đầu (9 số từ 0-8):", font=self.custom_font, bg='#F0F8FF').grid(row=0, column=0)
        self.initial_entry = tk.Entry(input_frame, width=20, font=self.custom_font)
        self.initial_entry.grid(row=0, column=1, padx=5)
        self.initial_entry.insert(0, "0,2,4,5,1,6,3,7,8")
        tk.Label(input_frame, text="Trạng thái mục tiêu:", font=self.custom_font, bg='#F0F8FF').grid(row=1, column=0)
        self.goal_entry = tk.Entry(input_frame, width=20, font=self.custom_font)
        self.goal_entry.grid(row=1, column=1, padx=5)
        self.goal_entry.insert(0, "1,2,3,4,5,6,7,8,0")
        tk.Button(input_frame, text="Cập nhật", command=self.update_states, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=2, rowspan=2, padx=5)

        tk.Label(main_frame, text="Trạng thái ban đầu", font=('Helvetica', 14, 'bold'), bg='#F0F8FF').grid(row=1, column=0, columnspan=3)
        self.start_labels = [[tk.Label(main_frame, text=str(puzzle.initial_state[r][c]) if puzzle.initial_state[r][c] != 0 else '',
                                      font=('Helvetica', 18), width=4, height=2, relief='solid', bg='#D3E5FF')
                             for c in range(3)] for r in range(3)]
        for r in range(3):
            for c in range(3):
                self.start_labels[r][c].grid(row=r+2, column=c, padx=2, pady=2)

        tk.Label(main_frame, text="Quá trình giải", font=('Helvetica', 14, 'bold'), bg='#F0F8FF').grid(row=1, column=4, columnspan=3)
        self.tiles = [[tk.Label(main_frame, text='', font=('Helvetica', 18), width=4, height=2, relief='solid', bg='#E5FFD3')
                       for c in range(3)] for r in range(3)]
        for r in range(3):
            for c in range(3):
                self.tiles[r][c].grid(row=r+2, column=c+4, padx=2, pady=2)

        button_frame = tk.Frame(main_frame, bg='#F0F8FF')
        button_frame.grid(row=5, column=0, columnspan=7, pady=10)
        algorithms = [
            ("BFS", self.run_bfs), ("DFS", self.run_dfs), ("IDS", self.run_ids), ("UCS", self.run_ucs),
            ("Greedy", self.run_greedy), ("A*", self.run_astar), ("IDA*", self.run_idastar),
            ("Simple HC", self.run_simple_hc), ("Steepest HC", self.run_steepest_hc), ("Random HC", self.run_stochastic_hc),
            ("SA", self.run_simulated_annealing), ("Beam", self.run_beam_search), ("GA", self.run_genetic_algorithm),
            ("AND-OR", self.run_and_or_search), ("Belief AND-OR", self.run_belief_and_or), ("Partial Obs", self.run_partial_obs),
            ("Backtracking", self.run_backtracking), ("Forward Check", self.run_forward_checking), ("Min-Conflicts", self.run_min_conflicts), ("Q-Learning", self.run_q_learning)
        ]
        for i, (text, cmd) in enumerate(algorithms):
            tk.Button(button_frame, text=text, command=cmd, width=13, font=self.custom_font, bg='#ADD8E6').grid(row=i//5, column=i%5, padx=3, pady=3)

        control_frame = tk.Frame(main_frame, bg='#F0F8FF')
        control_frame.grid(row=6, column=0, columnspan=7, pady=5)
        tk.Button(control_frame, text="<< Lùi", command=self.prev_step, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=0, padx=5)
        tk.Button(control_frame, text="Tiến >>", command=self.next_step, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=1, padx=5)
        tk.Button(control_frame, text="Phát", command=self.play_solution, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=2, padx=5)
        tk.Button(control_frame, text="Dừng", command=self.stop_play, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=3, padx=5)
        tk.Button(control_frame, text="Reset", command=self.reset_board, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=4, padx=5)
        tk.Button(control_frame, text="Hiển thị lịch sử", command=self.show_history, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=5, padx=5)

        self.info_label = tk.Label(main_frame, text="Chưa chạy thuật toán.", font=('Helvetica', 12), fg='green', bg='#F0F8FF')
        self.info_label.grid(row=7, column=0, columnspan=7, pady=5)

    def update_states(self):
        try:
            initial = [int(x) for x in self.initial_entry.get().split(',')]
            goal = [int(x) for x in self.goal_entry.get().split(',')]
            if len(initial) != 9 or len(goal) != 9:
                raise ValueError("Phải nhập đúng 9 số.")
            if set(initial) != set(range(9)) or set(goal) != set(range(9)):
                raise ValueError("Các số phải từ 0 đến 8, không trùng lặp.")
            self.puzzle.initial_state = [initial[i:i+3] for i in range(0, 9, 3)]
            self.puzzle.goal_state = [goal[i:i+3] for i in range(0, 9, 3)]
            self.original_state = [row[:] for row in self.puzzle.initial_state]
            for r in range(3):
                for c in range(3):
                    val = self.puzzle.initial_state[r][c]
                    self.start_labels[r][c]['text'] = str(val) if val != 0 else ''
            self.reset_board()
            if not self.puzzle.is_solvable():
                messagebox.showwarning("Cảnh báo", "Bài toán không thể giải được!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi nhập liệu: {str(e)}")

    def update_grid(self):
        if self.solution_steps and self.current_step < len(self.solution_steps):
            state = self.solution_steps[self.current_step]
            for r in range(3):
                for c in range(3):
                    val = state[r][c]
                    self.tiles[r][c]['text'] = str(val) if val != 0 else ''
                    self.tiles[r][c]['bg'] = '#FFA07A' if val == 0 else '#E5FFD3'

    def create_belief_window(self, algo_name):
        if self.belief_window and self.belief_window.winfo_exists():
            self.belief_window.destroy()
        self.belief_window = tk.Toplevel(self.root)
        self.belief_window.title(f"{algo_name} - 8-Puzzle Solver")
        self.belief_window.geometry("1000x600")
        self.belief_window.configure(bg='#F0F8FF')

        belief_frame = tk.Frame(self.belief_window, bg='#F0F8FF')
        belief_frame.pack(pady=10, padx=10)
        tk.Label(belief_frame, text=f"{algo_name} Belief States", font=('Helvetica', 14, 'bold'), bg='#F0F8FF').grid(row=0, column=0, columnspan=4)
        self.belief_labels = [[None for _ in range(3)] for _ in range(3)]
        for i in range(3):
            tk.Label(belief_frame, text=f"Belief {i+1}", font=self.custom_font, fg='green', bg='#F0F8FF').grid(row=1, column=i, padx=5)
            self.belief_labels[i] = [[tk.Label(belief_frame, text='', font=('Helvetica', 18), width=4, height=2, relief='solid', bg='#E0FFFF')
                                     for c in range(3)] for r in range(3)]
            for r in range(3):
                for c in range(3):
                    self.belief_labels[i][r][c].grid(row=r+2, column=i*3+c, padx=2, pady=2)

        solution_frame = tk.Frame(self.belief_window, bg='#F0F8FF')
        solution_frame.pack(pady=10, padx=10)
        tk.Label(solution_frame, text="Quá trình giải", font=('Helvetica', 14, 'bold'), bg='#F0F8FF').grid(row=0, column=0, columnspan=3)
        self.solution_tiles = [[tk.Label(solution_frame, text='', font=('Helvetica', 18), width=4, height=2, relief='solid', bg='#E5FFD3')
                               for c in range(3)] for r in range(3)]
        for r in range(3):
            for c in range(3):
                self.solution_tiles[r][c].grid(row=r+1, column=c+1, padx=2, pady=2)

        control_frame = tk.Frame(self.belief_window, bg='#F0F8FF')
        control_frame.pack(pady=5)
        tk.Button(control_frame, text="<< Lùi", command=self.prev_step_belief, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=0, padx=5)
        tk.Button(control_frame, text="Tiến >>", command=self.next_step_belief, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=1, padx=5)
        tk.Button(control_frame, text="Phát", command=self.play_solution_belief, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=2, padx=5)
        tk.Button(control_frame, text="Dừng", command=self.stop_play_belief, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=3, padx=5)
        tk.Button(control_frame, text="Đóng", command=self.close_belief_window, font=self.custom_font, bg='#ADD8E6').grid(row=0, column=4, padx=5)

        self.info_label_belief = tk.Label(self.belief_window, text="Chưa chạy thuật toán.", font=('Helvetica', 12), fg='green', bg='#F0F8FF')
        self.info_label_belief.pack(pady=5)

    def update_belief_grid(self):
        if self.solution_steps and self.current_step < len(self.solution_steps):
            state = self.solution_steps[self.current_step]
            for r in range(3):
                for c in range(3):
                    val = state[r][c]
                    self.solution_tiles[r][c]['text'] = str(val) if val != 0 else ''
                    self.solution_tiles[r][c]['bg'] = '#FFA07A' if val == 0 else '#E5FFD3'

    def update_belief_states(self, belief_set):
        if belief_set and self.belief_window and self.belief_window.winfo_exists():
            belief_list = sorted(belief_set, key=self.puzzle.heuristic)[:3]  # Lấy 3 trạng thái tốt nhất
            # Đảm bảo hiển thị 3 belief states, lặp lại nếu thiếu
            while len(belief_list) < 3:
                belief_list.append(belief_list[0])  # Lặp lại belief đầu tiên nếu thiếu
            for i in range(3):
                state = list(map(list, belief_list[i]))
                for r in range(3):
                    for c in range(3):
                        val = state[r][c]
                        self.belief_labels[i][r][c]['text'] = str(val) if val != 0 else ''
                        self.belief_labels[i][r][c]['bg'] = '#FFA07A' if val == 0 else '#E0FFFF'

    def solve_belief(self, algorithm, algo_name):
        if not self.puzzle.is_solvable():
            self.info_label_belief.config(text="Bài toán không thể giải được.", fg='red')
            return
        solution, elapsed_time, state_count, belief_set = algorithm()
        self.update_belief_states(belief_set)
        steps = len(solution) - 1 if solution else 0
        self.performance_data.append({
            'name': algo_name,
            'steps': steps,
            'time': elapsed_time,
            'states': state_count
        })
        if solution:
            self.solution_steps = solution
            self.current_step = 0
            self.update_belief_grid()
            self.info_label_belief.config(
                text=f"Thuật toán: {algo_name} | Bước: {steps} | Thời gian: {elapsed_time:.4f}s | Trạng thái duyệt: {state_count}",
                fg='green'
            )
            self.history.append(f"{algo_name}: {steps} bước, {elapsed_time:.4f}s, {state_count} trạng thái")
        else:
            self.solution_steps = []
            self.info_label_belief.config(text=f"{algo_name}: Không tìm thấy lời giải.", fg='red')

    def next_step_belief(self):
        if self.solution_steps and self.current_step < len(self.solution_steps) - 1 and self.belief_window:
            self.current_step += 1
            self.update_belief_grid()

    def prev_step_belief(self):
        if self.solution_steps and self.current_step > 0 and self.belief_window:
            self.current_step -= 1
            self.update_belief_grid()

    def play_solution_belief(self):
        if not self.solution_steps or self.is_playing or not self.belief_window:
            return
        self.is_playing = True
        self.current_step = 0
        self.update_belief_grid()
        def play():
            if self.is_playing and self.current_step < len(self.solution_steps) - 1:
                self.next_step_belief()
                self.belief_window.after(500, play)
            else:
                self.is_playing = False
        self.belief_window.after(500, play)

    def stop_play_belief(self):
        self.is_playing = False

    def close_belief_window(self):
        if self.belief_window and self.belief_window.winfo_exists():
            self.belief_window.destroy()
        self.belief_window = None
        self.solution_steps = None
        self.current_step = 0

    def solve(self, algorithm, algo_name):
        if not self.puzzle.is_solvable():
            self.info_label.config(text="Bài toán không thể giải được.", fg='red')
            return
        solution, elapsed_time, state_count = algorithm()
        steps = len(solution) - 1 if solution else 0
        self.performance_data.append({
            'name': algo_name,
            'steps': steps,
            'time': elapsed_time,
            'states': state_count
        })
        if solution:
            self.solution_steps = solution
            self.current_step = 0
            self.info_label.config(
                text=f"Thuật toán: {algo_name} | Bước: {steps} | Thời gian: {elapsed_time:.4f}s | Trạng thái duyệt: {state_count}",
                fg='green'
            )
            self.history.append(f"{algo_name}: {steps} bước, {elapsed_time:.4f}s, {state_count} trạng thái")
        else:
            self.solution_steps = []
            self.info_label.config(text=f"{algo_name}: Không tìm thấy lời giải.", fg='red')

    def next_step(self):
        if self.solution_steps and self.current_step < len(self.solution_steps) - 1:
            self.current_step += 1
            self.update_grid()

    def prev_step(self):
        if self.solution_steps and self.current_step > 0:
            self.current_step -= 1
            self.update_grid()

    def play_solution(self):
        if not self.solution_steps or self.is_playing:
            return
        self.is_playing = True
        self.current_step = 0
        self.update_grid()
        def play():
            if self.is_playing and self.current_step < len(self.solution_steps) - 1:
                self.next_step()
                self.root.after(500, play)
            else:
                self.is_playing = False
        self.root.after(500, play)

    def stop_play(self):
        self.is_playing = False

    def reset_board(self):
        self.puzzle.initial_state = [row[:] for row in self.original_state]
        self.solution_steps = None
        self.current_step = 0
        self.is_playing = False
        self.info_label.config(text="Đã reset về trạng thái ban đầu.", fg='blue')
        for r in range(3):
            for c in range(3):
                val = self.puzzle.initial_state[r][c]
                self.start_labels[r][c]['text'] = str(val) if val != 0 else ''
                self.tiles[r][c]['text'] = ''
                self.tiles[r][c]['bg'] = '#E5FFD3'

    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Lịch sử giải")
        history_window.geometry("400x300")
        history_window.configure(bg='#F0F8FF')

        tk.Label(history_window, text="Lịch sử giải:", font=('Helvetica', 14, 'bold'), bg='#F0F8FF').pack(pady=5)
        history_text = tk.Text(history_window, height=10, width=50, font=('Helvetica', 10), bg='#E6E6FA')
        history_text.pack(pady=5)
        history_text.insert(tk.END, "\n".join(self.history[-10:]))
        history_text.config(state='disabled')

        tk.Button(history_window, text="Đóng", command=history_window.destroy, font=self.custom_font, bg='#ADD8E6').pack(pady=5)

    def run_and_or_search(self):
        self.solve(self.puzzle.and_or_search, "AND-OR")

    def run_bfs(self): self.solve(self.puzzle.solve_bfs, "BFS")
    def run_dfs(self): self.solve(self.puzzle.solve_dfs, "DFS")
    def run_ids(self): self.solve(self.puzzle.solve_ids, "IDS")
    def run_ucs(self): self.solve(self.puzzle.solve_ucs, "UCS")
    def run_greedy(self): self.solve(self.puzzle.solve_greedy, "Greedy")
    def run_astar(self): self.solve(self.puzzle.solve_astar, "A*")
    def run_idastar(self): self.solve(self.puzzle.solve_idastar, "IDA*")
    def run_simple_hc(self): self.solve(self.puzzle.solve_simple_hill_climbing, "Simple HC")
    def run_steepest_hc(self): self.solve(self.puzzle.solve_steepest_ascent_hill_climbing, "Steepest HC")
    def run_stochastic_hc(self): self.solve(self.puzzle.solve_stochastic_hill_climbing, "Random HC")
    def run_simulated_annealing(self): self.solve(self.puzzle.solve_simulated_annealing, "SA")
    def run_beam_search(self): self.solve(lambda: self.puzzle.solve_beam_search(beam_width=3), "Beam")
    def run_genetic_algorithm(self): self.solve(self.puzzle.solve_genetic_algorithm, "GA")
    def run_belief_and_or(self):
        self.create_belief_window("Belief AND-OR")
        self.solve_belief(self.puzzle.solve_belief_state_and_or_search, "Belief AND-OR")
    def run_partial_obs(self):
        self.create_belief_window("Partial Obs")
        self.solve_belief(self.puzzle.solve_partial_observable_search, "Partial Obs")
    def run_backtracking(self): self.solve(self.puzzle.solve_backtracking, "Backtracking")
    def run_forward_checking(self): self.solve(self.puzzle.solve_forward_checking, "Forward Check")
    def run_min_conflicts(self): self.solve(self.puzzle.solve_min_conflicts, "Min-Conflicts")
    def run_q_learning(self): self.solve(self.puzzle.solve_q_learning, "Q-Learning")

if __name__ == "__main__":
    initial_state = [[0, 2, 4], [5, 1, 6], [3, 7, 8]]
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    root = tk.Tk()
    root.title("8-Puzzle Solver - Advanced")
    root.geometry("800x600")
    puzzle = EightPuzzle(initial_state, goal_state)
    app = EightPuzzleGUI(root, puzzle)
    root.mainloop()