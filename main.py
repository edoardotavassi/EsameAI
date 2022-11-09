import maze_generator as mg
import computer_vision_maze as cvm
import treasure_maze as tm
import os


OKGREEN = "\033[92m"
WARNING = "\033[93m"
OKCYAN = "\033[96m"
CLEAN = "\033[0m"

def printRed(string):
    print(WARNING + string + CLEAN)
    
def printGreen(string):
    print(OKGREEN + string + CLEAN)

def printCyan(string):
    print(OKCYAN + string + CLEAN)

map = []
printGreen("Vuoi generare un labirinto casuale o caricare un labirinto da file?")
printCyan("1. Genera labirinto casuale")
printCyan("2. Carica labirinto da file")
printCyan("3. Carica l'ultimo labirinto generato (se disponibile)")
maze_type = input("Scelta: ")
if maze_type == "1":
    if os.path.exists("./img/maze.jpg"):
        os.remove("./img/maze.jpg")
        

    printGreen("Dimensione del labirinto? (2-7)")
    flag = True
    while flag:
        choice = input("Scelta: ")
        choice = int(choice)
        print(choice)
        if choice<2 or choice>7:
            printRed("Dimensione scorretta (2-7)")
        else:
            flag = False
            printGreen("Generazione labirinto...")
    maze = mg.Maze(choice, choice)
    maze.visualize()
    visual = mg.Visualizer(maze, "maze")
    visual.generate()
    visual.saveImage("img/")
    analyzer = cvm.ImageAnalysis("./img/maze.jpg")
    map = analyzer.digital_maze_detection()
    
elif maze_type == "2":
    print("Inserisci il nome del file (senza estensione)")
    string = "./img/" + input("Nome: ") + ".jpg"
    if os.path.exists(string):
        analyzer = cvm.ImageAnalysis(string)
        map = analyzer.hand_maze_detection()
    else:
        print("Errore: nessun labirinto trovato")
        exit()
    

elif maze_type == "3":
    if os.path.exists("./img/maze.jpg"):
        analyzer = cvm.ImageAnalysis("./img/maze.jpg")
        map = analyzer.digital_maze_detection()
    else:
        print("Errore: nessun labirinto trovato")
        exit()
else:
    exit()
    

printGreen("Quanti tesori vuoi trovare?")
printGreen("Scrivi 0 per trovarli tutti")
choice = int(input("Scelta: "))
if choice == 0:
    problem = tm.TreasureMaze(map)
else:
    problem = tm.TreasureMaze(map, True, choice)


printGreen("Quale algoritmo vuoi utilizzare?")
printCyan("1. A*")
printCyan("2. BFS")
printCyan("3. DFS")
choice = int(input("Scelta: "))
while choice<1 or choice>3:
    printRed("Scelta errata")
    printCyan("1. A*")
    printCyan("2. BFS")
    printCyan("3. DFS")
    choice = int(input("Scelta: "))
    
if choice == 1:
    printGreen("Algoritmo A*")
    title ="A*"
    (time, iterations, node)= tm.astar_search_graph(problem, display=True)
elif choice == 2:
    printGreen("Algoritmo BFS")
    title ="Breath First Search"
    (time, iterations, node)= tm.breadth_first_graph_search(problem)
elif choice == 3:  
    printGreen("Algoritmo DFS")
    title ="Depth First Search"
    (time, iterations, node)= tm.depth_first_graph_search(problem)
    

node_path = node.path()
print(node_path)
printGreen("_________________________________________________________")
printRed(title)
print("Tempo di esecuzione: ", time, end="")
print(" s")
print("Numero di iterazioni: ", iterations)
printGreen("_________________________________________________________")
printCyan("Usa le frecce per muoverti tra gli stati del labirinto")
printCyan("Premi Spazio per far partire l'animazione")
printCyan("Chiudi la finestra per terminare l'esecuzione")
printRed("NB: Non premere Spazio durante un'altra animazione \n(l'operzione si ripeterà al termine dell'animazione in corso)")
tm.TreasureGame(node_path, title)