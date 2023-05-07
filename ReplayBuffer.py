from GraphState import GraphState


class ReplayBuffer:
    def __init__(self, max_iterations, size=20, invalidate_duration=4, num_verts=30, num_edges=15, delta_e=10, prepopulate_start=True):
        self.max_iterations = max_iterations
        self.num_verts = num_verts
        self.num_edges = num_edges
        self.delta_e = delta_e
        self.prepopulate_start = prepopulate_start
        self.buffer = [GraphState(regenerate_graphs=False, num_verts=self.num_verts, num_edges=self.num_edges, delta_e=self.delta_e, prepopulate_start=self.prepopulate_start)
                       for _ in range(size)]
        self.invalidate_duration = invalidate_duration

        self.num_iterations = 0
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_iterations > self.max_iterations:
            raise StopIteration
        else:
            self.num_iterations += 1

            if self.num_iterations % (self.size * self.invalidate_duration) == 0:
                self.buffer = [GraphState(regenerate_graphs=False, num_verts=self.num_verts, num_edges=self.num_edges, delta_e=self.delta_e, prepopulate_start=self.prepopulate_start)
                               for _ in range(self.size)]

            return self.buffer[(self.num_iterations - 1) % len(self.buffer)]
