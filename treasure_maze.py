from search import *
from time import sleep
import math
import tkinter as tk
import timeit

game_map = [
"X","T","2","4",
"X","X","1","X",
"T","4","X","4",
"4","3","S","1",
]

"""game_map = [
"S","T","2","4",
"X","X","1","X",
"T","4","X","4",
"4","3","X","1",
]"""


"""game_map = [
"S","T","2","4","2","X",
"X","X","1","X","1","X",
"T","4","X","4","X","X",
"4","3","X","1","3","2",
"4","3","X","1","3","2",
"3","4","X","X","X","T",
]"""

"""game_map = [
"S","T","2","4","2","X","1", "3", "3",
"X","X","1","X","1","X","X", "2", "X",
"T","4","X","4","X","X","2", "2", "3",
"4","3","X","1","3","2","X", "3", "X",
"4","3","X","1","3","2","2", "X", "3",
"3","4","X","X","X","T","4", "2", "X",
"3","4","X","X","X","T","X", "X", "2",
"4","3","X","1","3","2","4", "X", "3",
"T","4","X","4","X","X","1", "2", "1"
]"""




def printState(state, goal):
    n = int(math.sqrt(len(state)))
    print("------------------")
    count =1
    for i in state:
        if i[1]:
            print("\033[92m", end="")
            print(i, end=" ")
            print('\033[0m', end="")
        else:
            print(i, end=" ")
        if count == n:
            print("\n")
            count = 0
        count +=1
    print("GOAL: ", end="")
    print(goal)
    print("------------------")
    


class TreasureMaze(Problem):
    def cost(self, cella):
        # Return the cost of the cell
        if cella == "1":
            return 1
        elif cella == "2":
            return 2
        elif cella == "3":
            return 3
        elif cella == "4":
            return 4
        elif cella == "X":
            return 5
        elif cella == "T":
            return 0
        elif cella == "S":
            return 0
        elif cella == "O":
            return 0

    def goal_calc(self, initial):
        # find the locations of treasures
        count = 0
        goal = set()
        for (i,_) in initial:
            if i == "T":
                goal.add(count)
            count += 1
        return goal
    
    def goal_number(self, initial):
        # Count the number of treasures
        count = 0
        for i in initial:
            if i == "T":
              count += 1
        return count

    def initial_state_calc(self, initial):
        # Find the starting position
        new_initial = list()
        for i in initial:
            if i == "S":
                new_initial.append(("S", True))
            else:
                new_initial.append((i, False))
        return new_initial

    def get_agent_location(self, state):
        """Return the location of the agent in the given state"""
        count = 0
        for (_, agent_bool) in state:
            if agent_bool:
                return count
            count += 1
        return 0

    def __init__(self, initial, easy_goal=False, goal_n=0):
        """Specifica lo stato iniziale, i possibili stati goal
        e il numero di righe e di colonne della matrice di input di grandezza mxn"""
        self.n = int(math.sqrt(len(initial)))
        if easy_goal and goal_n < self.goal_number(initial):
            goal = goal_n
        else:
            goal = self.goal_number(initial)
        super().__init__(self.initial_state_calc(initial), self.n, goal)
        

    def actions(self, state):
        agent=self.get_agent_location(state)
        movement = { "L": -1, "R": +1, "U": -self.n, "D": +self.n}
        moves_list = ["L", "R", "U", "D"]
        actions_list = list()
        
        if agent % self.n == 0:
            moves_list.remove("L")
        if agent < self.n:
            moves_list.remove("U")
        if agent % self.n == self.n - 1:
            moves_list.remove("R")
        if agent >= (self.n * (self.n - 1)):
            moves_list.remove("D")
        
        for move in moves_list:
            actions_list.append((move, self.cost(state[agent + movement[move]][0])))
                    
        return actions_list

    def result(self, state, action):
        agent=self.get_agent_location(state)
        new_state = state.copy()
        movement = { "L": -1, "R": +1, "U": -self.n, "D": +self.n}
        action_cell_index = agent + movement[action[0]]
        #print(str(action_cell_index) + " "+ str(agent))
        
        #Wall
        if new_state[action_cell_index][0] == "X":
            new_state[action_cell_index] = ("1", False)
        
        #Walkable
        else:
            #Treasure
            if new_state[action_cell_index][0] == "T":
                cell_text = "O"
            else:
                cell_text = new_state[action_cell_index][0]
                    
            new_state[action_cell_index] = (cell_text, True)
            new_state[agent] = (new_state[agent][0], False)
        
        return new_state

    def goal_test(self, state):
        count=0
        for (i, _) in state:
            if i == "O":
                count +=1
            if count == self.goal:
                return True
        return False
                

    def path_cost(self, c, action):
        return c + action[1]
    
    def h(self, node):
        state = node.state
        agent=self.get_agent_location(state)
        positions = list()
        for i in self.goal_calc(state):
                positions.append((int(i)%self.n, math.floor(int(i)/self.n) ))
        
        agent=(agent%self.n, math.floor(agent/self.n))
        
        heuristic = []
        for i in positions:
            heuristic.append(abs(i[0] - agent[0])+abs(i[1] - agent[1]))
        if heuristic == []:
            return 0
        return min(heuristic)



problem = TreasureMaze(game_map)
#print(problem.actions(problem.initial))
"""node = Node(problem.initial)
print(node )
print(problem.h(node))"""

#printState(problem.result(problem.initial, ('R', 1)),4, [])
"""
print(problem.actions(problem.initial))
problem.result(problem.initial, ('R', 1))
print(problem.goal)
print(problem.goal_test())
print(problem.path_cost(0, ('R', 1)))"""


"""state = problem.initial
print("Initial state:")
cost=0
while not problem.goal_test(state):
    printState(state, problem.n, problem.goal)
    print("Costo: " + str(cost))
    action_list = problem.actions(state)
    move=action_list[0]
    h=problem.h(state, move[0])
    for i in action_list:
        h2= problem.h(state, i[0])
        if h>h2:
            move=i
            h=h2
    cost=problem.path_cost(cost, move)
    state = problem.result(state, move)
    time.sleep(1)
printState(state, problem.n, problem.goal)
print("Costo: " + str(cost))
"""


"""frontier = [Node(problem.initial)]
print(frontier)
explored = set()
node = frontier.pop()
print(problem.goal_test())
explored.add(tuple(node.state))
#print(node.expand(problem))
print("EXPAND")
frontier.extend(child for child in node.expand(problem) if tuple(child.state) not in explored and child not in frontier)
print(frontier)"""

"""________________________________________________"""


def depth_first_graph_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    start=timeit.default_timer()
    iteration=1
    frontier = [(Node(problem.initial))]  # Stack
    explored = set()
    while frontier:
        node = frontier.pop()
        """print("\033[92m"+str(node.action) +'\033[0m')
        print(node)"""
        if problem.goal_test(node.state):
            return (timeit.default_timer()-start, iteration, node)
        explored.add(tuple(node.state))
        frontier.extend(
            child
            for child in node.expand(problem)
            if tuple(child.state) not in explored and child not in frontier
        )
        iteration+=1
        
    return None

def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    start=timeit.default_timer()
    iteration=1
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return (timeit.default_timer()-start, iteration, node)
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(tuple(node.state))
        for child in node.expand(problem):
            if tuple(child.state) not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return (timeit.default_timer()-start, iteration, child)
                frontier.append(child)
        iteration += 1
    
    return None

def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    
    f = memoize(f, "f")
    node = Node(problem.initial)
    frontier = PriorityQueue("min", f) 
    frontier.append(node)
    explored = set()
    start=timeit.default_timer()
    iteration =0
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return (timeit.default_timer()-start, iteration, node)
        
        explored.add(node)
        for child in node.expand(problem):
            if child not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
        iteration+=1
        
    return None


"""________________________________________________"""

def astar_search_graph(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    (time, iteration, node) = best_first_graph_search(problem, lambda n: n.path_cost + h(n))
    return (time, iteration, node)

"""________________________________________________"""


(time, iterations, node)= depth_first_graph_search(problem)
print(node)
print(time)
print(iterations)
#print(node)
(time, iterations, node) = breadth_first_graph_search(problem)

#node_path = node.path()
#print (node_path)
print(node)
print(time)
print(iterations)

(time, iterations, node) = astar_search_graph(problem)

print(node)
print(time)
print(iterations)




CELL_COLORS = {
    "1": "#bfff00",
    "2": "#bfff00",
    "3": "#bfff00",
    "4": "#bfff00",
    "X": "#595959",
    "T": "#fff200",
    "O": "#4fffff",
    "S": "#ff7a7a",
    "A": "#ff0000",
}

EMPTY_COLOR = "#ffffff"
LETTERS_COLOR = "#000000"
GRID_COLOR = "#000000"

LETTERS_FONT = ("Helvetica", 55, "bold")

class Game(tk.Frame):
    def __init__(self, node_path, node_index=0):
        tk.Frame.__init__(self)
        self.grid()
        self.master.title("Treasure Maze")
        self.node_path=node_path
        self.node_index=node_index
        self.dimensions= int(math.sqrt(len(self.node_path[0].state)))
        self.main_grid=tk.Frame(
            self, bg=GRID_COLOR, bd=3, width=600, height=600
        )
        self.main_grid.grid(pady=(100,0))
        self.make_GUI()
        self.start_game(node_path[self.node_index])
        
        self.master.bind("<Left>", self.left)
        self.master.bind("<Right>", self.right)
        self.master.bind("<space>", self.space)
        self.mainloop()
        
    def make_GUI(self):
        self.cells=[]
        for i in range(self.dimensions):
            row=[]
            for j in range(self.dimensions):
                cell_frame=tk.Frame(
                    self.main_grid,
                    bg=EMPTY_COLOR,
                    width=600/self.dimensions,
                    height=600/self.dimensions
                )
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_number=tk.Label(self.main_grid, bg=EMPTY_COLOR)
                cell_number.grid(row=i, column=j)
                cell_data={"frame": cell_frame, "number": cell_number}
                row.append(cell_data)
            self.cells.append(row)
            
        #Path cost
        cost_frame = tk.Frame(self)
        cost_frame.place(relx=0.5, y=40, anchor="center")
        tk.Label(
            cost_frame,
            text="Path Cost",
            font=("Helvetica", 20, "bold"),
        ).grid(row=0)
        self.cost_label = tk.Label(cost_frame, text="0", font=("Helvetica", 20))
        self.cost_label.grid(row=1)
        
    def read_state(self, node):
        for row in range (0,self.dimensions):
            for col in range(0,self.dimensions):
                self.matrix[row][col] =node.state[row*self.dimensions+col]
        self.cost = node.path_cost
            
    def color_matrix(self):
        for row in range (0,self.dimensions):
            for col in range(0,self.dimensions):
                if(self.matrix[row][col][1]):
                    self.cells[row][col]["frame"].configure(bg=CELL_COLORS["A"])
                    self.cells[row][col]["number"].configure(
                        bg=CELL_COLORS["A"],
                        fg=LETTERS_COLOR,
                        font=LETTERS_FONT,
                        text=self.matrix[row][col][0])
                else:
                    self.cells[row][col]["frame"].configure(bg=CELL_COLORS[self.matrix[row][col][0]])
                    self.cells[row][col]["number"].configure(
                        bg=CELL_COLORS[self.matrix[row][col][0]],
                        fg=LETTERS_COLOR,
                        font=LETTERS_FONT,
                        text=self.matrix[row][col][0])
        self.cost_label.configure(text=str(self.cost))
        
        
    def start_game(self, node):
        self.matrix = [[0] * self.dimensions for _ in range(self.dimensions)]
        self.cost = 0    
        self.read_state(node)
        self.color_matrix()
        
    def update_GUI(self):
        self.read_state(self.node_path[self.node_index])
        self.color_matrix()
        self.update_idletasks()
        
        
    def left(self, event):
        if self.node_index > 0:
            self.node_index -= 1
            self.update_GUI()
        
    def right(self, event):
        if self.node_index < len(self.node_path) - 1:
            self.node_index += 1
            self.update_GUI()
            
    def space(self, event):
        self.node_index=0
        self.cost_label.configure(fg="red")
        while self.node_index < len(self.node_path) - 1:
            self.node_index += 1
            self.update_GUI()
            sleep(0.4)
        self.cost_label.configure(fg="white")
        self.node_index=0
        self.update_GUI()
            

Game(node_path)