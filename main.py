import maze_generator as mg
import computer_vision_maze as cvm
import treasure_maze as tm
import os


map = []
print("Vuoi generare un labirinto casuale o caricare un labirinto da file?")
print("1. Genera labirinto casuale")
print("2. Carica labirinto da file")
maze_type = input("Scelta: ")
if maze_type == "1":
    if os.path.exists("./img/maze.jpg"):
        os.remove("./img/maze.jpg")
        

    print("Dimensione del labirinto? (2-7)")
    flag = True
    while flag:
        choice = input("Scelta: ")
        choice = int(choice)
        print(choice)
        if choice<2 or choice>7:
            print("Dimensione scorretta (2-7)")
        else:
            flag = False
            print ("Generazione labirinto...")
    maze = mg.Maze(choice, choice)
    maze.visualize()
    visual = mg.Visualizer(maze, "maze")
    visual.generate()
    visual.saveImage("img/")
    analyzer = cvm.ImageAnalysis("./img/maze.jpg")
    map = analyzer.digital_maze_detection()
    
elif maze_type == "2":
    exit()
else:
    exit()
    
    
problem = tm.TreasureMaze(map)

print("Quale algoritmo vuoi utilizzare?")
print("1. A*")
print("2. BFS")
print("3. DFS")
choice = int(input("Scelta: "))
while choice<1 or choice>3:
    print("Scelta errata")
    print("1. A*")
    print("2. BFS")
    print("3. DFS")
    choice = int(input("Scelta: "))
    
if choice == 1:
    print("Algoritmo A*")
elif choice == 2:
    print("Algoritmo BFS")
elif choice == 3:  
    print("Algoritmo DFS")
    (time, iterations, node)= tm.depth_first_graph_search(problem)
    

node_path = node.path()
print(node_path)
print("Tempo di esecuzione: ", time)
print("Numero di iterazioni: ", iterations)
tm.Game(node_path)