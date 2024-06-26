import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """

    manDist = 0

    for i in range(9):

        if from_state[i] != 0:

            ind = to_state.index(from_state[i])
            manDist += abs(i//3 - ind//3) + abs(i%3 - ind%3)
            
    return manDist

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """

    succ_states = []

    moves = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    for i in range(9):

        if state[i] == 0:

            row_empty, col_empty = divmod(i, 3)

            for row_move, col_move in moves:
                row_neigh, col_neigh = row_empty + row_move, col_empty + col_move

                if 0 <= row_neigh < 3 and 0 <= col_neigh < 3:

                    succ = state.copy()
                    succ[i], succ[row_neigh * 3 + col_neigh] = succ[row_neigh * 3 + col_neigh], succ[i]

                    if succ != state:

                        succ_states.append(succ)

    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

    pq = [(get_manhattan_distance(state), state, (0, get_manhattan_distance(state), -1))]
    visitedGrid = {tuple(state)}
    max_length = 0
    state_info_list = []
    g_dict = {}

    while pq:

        _, current_state, (g, h, parent_index) = heapq.heappop(pq)
        state_info_list.append((current_state, h, g, parent_index))

        visitedGrid.add(tuple(current_state))

        if current_state == goal_state:
            break

        succ_states = get_succ(current_state)
        
        for succ_state in succ_states:

            if tuple(succ_state) in visitedGrid:
                continue

            key = "".join(map(str, succ_state))
            succ_dist = get_manhattan_distance(succ_state)
            curr_ind = len(state_info_list) -1
            curr_g = g + 1 + succ_dist

            if key in g_dict:

                if g_dict[key] <= curr_g:
                    continue

            g_dict[key] = curr_g

            heapq.heappush(pq, (g + 1 + succ_dist, succ_state, (g + 1, succ_dist, curr_ind)))
            max_length = max(max_length, len(pq))

    path = []
    curr_ind = len(state_info_list) - 1

    while curr_ind != -1:

        state_info = state_info_list[curr_ind]
        path.append((state_info[0], state_info[1], state_info[2]))
        curr_ind = state_info[3]
    
    state_info_list = reversed(path)

    # This is a format helper.
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute max length
    # it can help to avoid any potential format issue.
    for state_info in state_info_list:
        current_state = state_info[0]
        h = state_info[1]
        move = state_info[2]
        print(current_state, "h={}".format(h), "moves: {}".format(move))
    print("Max queue length: {}".format(max_length))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,3,6,7,0,0], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    # solve([2,5,1,4,0,6,7,0,3])
    # print()

    solve([4,3,0,5,1,6,7,2,0])
    print()

    print_succ([3, 4, 6, 0, 0, 1, 7, 2, 5])
    print_succ([6, 0, 0, 3, 5, 1, 7, 2, 4])
    print_succ([0, 4, 7, 1, 3, 0, 6, 2, 5])
    solve([3, 4, 6, 0, 0, 1, 7, 2, 5])
    solve([6, 0, 0, 3, 5, 1, 7, 2, 4])
    solve([0, 4, 7, 1, 3, 0, 6, 2, 5])