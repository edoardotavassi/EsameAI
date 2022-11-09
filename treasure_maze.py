from search import *
from time import sleep
import math
import tkinter as tk
import timeit


#Test Maps
"""game_map = [
"X","T","2","4",
"X","X","1","X",
"T","4","X","4",
"4","3","S","1",
]"""

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
        # Return the index of the agent in the state
        count = 0
        for (_, agent_bool) in state:
            if agent_bool:
                return count
            count += 1
        return 0

    def __init__(self, initial, easy_goal=False, goal_n=0):
        
        self.n = int(math.sqrt(len(initial)))
        
        #less goal
        if easy_goal and goal_n < self.goal_number(initial):
            goal = goal_n
        else:
            goal = self.goal_number(initial)
        super().__init__(self.initial_state_calc(initial), self.n, goal)
        

    def actions(self, state):
        # Return the actions that can be executed in the given state.
        # The result is a list in the form of [(action1, cost1),...(actionn, costn)])] 
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
        # Return the state that results from executing the given
        # action in the given state. The action must be one of
        # self.actions(state). 
        agent=self.get_agent_location(state)
        new_state = state.copy()
        movement = { "L": -1, "R": +1, "U": -self.n, "D": +self.n}
        action_cell_index = agent + movement[action[0]]
        
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
        # Return True if the state is a goal.
        count=0
        for (i, _) in state:
            if i == "O":
                count +=1
            if count == self.goal:
                return True
        return False
                

    def path_cost(self, c, action):
        # Return the cost of a path
        return c + action[1]
    
    def h(self, node):
        # Return the heuristic value for a given state
        # Manhattan distance for a multi-goal problem
        # The distance is calculated from the agent to the closest treasure
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
        
        #Goal test
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
    
    #if the initial state is the goal, return it
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

def best_first_graph_search(problem, f, display=False):
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
            if display:
                print(
                    len(explored),
                    "paths have been expanded and",
                    len(frontier),
                    "paths remain in the frontier",
                )
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

def astar_search_graph(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    (time, iteration, node) = best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)
    return (time, iteration, node)

"""________________________________________________"""


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

class TreasureGame(tk.Frame):
    def __init__(self, node_path, title, node_index=0):
        # Initialize the game
        tk.Frame.__init__(self)
        self.grid()
        self.master.title(title)
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
        # Make the GUI
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
        # Read the state of the game
        for row in range (0,self.dimensions):
            for col in range(0,self.dimensions):
                self.matrix[row][col] =node.state[row*self.dimensions+col]
        self.cost = node.path_cost
            
    def color_matrix(self):
        # Colors the cells of the game and add the cost
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
        # Initialize the game state
        self.matrix = [[0] * self.dimensions for _ in range(self.dimensions)]
        self.cost = 0    
        self.read_state(node)
        self.color_matrix()
        
    def update_GUI(self):
        # Update the GUI
        self.read_state(self.node_path[self.node_index])
        self.color_matrix()
        self.update_idletasks()
        
        
    def left(self, event):
        # Event for left arrow
        if self.node_index > 0:
            self.node_index -= 1
            self.update_GUI()
        
    def right(self, event):
        # Event for right arrow
        if self.node_index < len(self.node_path) - 1:
            self.node_index += 1
            self.update_GUI()
            
    def space(self, event):
        # Event for space
        # Play the game automatically (animation)
        self.node_index=0
        self.cost_label.configure(fg="red")
        while self.node_index < len(self.node_path) - 1:
            self.node_index += 1
            self.update_GUI()
            sleep(0.4)
        self.cost_label.configure(fg="white")
        self.node_index=0
        self.update_GUI()
            