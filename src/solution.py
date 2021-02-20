from csp import CSP, mrv, num_legal_values, dom_j_up, lcv, first_unassigned_variable, unordered_domain_values, no_inference, min_conflicts_value, random
from utils import argmin_random_tie, count, first, extend
import time
import copy
import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)



class RLFAP(CSP):

    def __init__(self, dom_path, var_path, ctr_path):
        # first we determine the domains
        self.dom = self.getDomains(dom_path)
        self.var = self.getVariables(var_path)
        self.ctr = self.getConstraints(ctr_path)
        variables = [x for x, dom_id in self.var]
        domains = self.assignDomains(self.var, self.dom)
        self.neighbors = self.getNeighbors(variables, self.ctr)
        # initialize based on the current data
        CSP.__init__(self, variables, domains,
                     self.neighbors, self.checkConstraints)

        # needed for the DOV heuristic dom/wdeg
        self.weights = {pair: 1 for pair in self.ctr.keys()}
        assert self.weights.keys() == self.ctr.keys()
        # needed for FC-CBJ
        self.conflicts = {x: set() for x in self.variables}
        # counter for constraints checks
        self.ctr_counter = 0
        self.assignment_cutoff = 10**6
        self.ctr_cutoff = 10**8

    def getDomains(self, dom_path) -> dict:
        """Return a dict (domain_id, values)"""
        domains = dict()
        with open(dom_path) as file:
            lines = file.readlines()
            for line in lines[1:]:
                elements = list(map(int, line.split()))
                domain_id = elements[0]
                domain_values = elements[2:]
                domains[domain_id] = domain_values

        return domains

    def getVariables(self, var_path) -> list:
        """Return a list of [variable, domain_id]"""
        variables = list()
        with open(var_path) as file:
            lines = file.readlines()
            variables = [list(map(int, pair.split())) for pair in lines[1:]]

        return variables

    def getConstraints(self, var_path) -> dict:
        """Return a dictionary {(x, y) : [(operation symbol, k)]}
            for example d[(x, y)] = [(">", 1)] => |x - y| > k
        """
        constraints = dict()
        with open(ctr_path) as file:
            lines = file.readlines()
            for line in lines[1:]:
                elem = list(line.split())
                x_id, y_id, operation, k = int(elem[0]), int(
                    elem[1]), elem[2], int(elem[3])
                if (x_id, y_id) not in constraints:
                    constraints[(x_id, y_id)] = [(operation, k)]
                else:
                    constraints[(x_id, y_id)] += [(operation, k)]

        return constraints

    def assignDomains(self, variables, domain) -> dict:
        """variables = list of [x, dom_id], domain is {dom_id:[values...]}
            Return dict of d[x] = [values...]
        """
        domains = {}
        for x, dom_id in variables:
            domains[x] = domain[dom_id].copy()

        return domains

    def checkConstraint(self, a, b, constraint) -> bool:
        # increase the constraint counter
        self.ctr_counter += 1

        operation, k = constraint
        if operation == ">":
            return abs(a-b) > k
        elif operation == "=":
            return abs(a-b) == k

        print("The constraint operation is unknown. Returning False")
        return False

    def checkConstraints(self, A, a, B, b) -> bool:
        """Check if the variable assignment satisfies constraints"""
        if (B, A) in self.ctr.keys():
            A, B = B, A
            a, b = b, a
        elif not ((A, B) in self.ctr.keys()):
            return True

        constraints_list = self.ctr[(A, B)]

        for constraint in constraints_list:
            if self.checkConstraint(a, b, constraint) == False:
                return False

        # if we reached this point then the variables satisfy all constraints
        return True

    def getNeighbors(self, variables, constraints) -> dict:
        """Get the neighbours dict"""
        neighbors = {}
        for x in variables:
            for y in variables:
                if x != y and ((x, y) in constraints):
                    if x not in neighbors:
                        neighbors[x] = [y]
                    else:
                        neighbors[x].append(y)

                    if y not in neighbors:
                        neighbors[y] = [x]
                    else:
                        neighbors[y].append(x)

        return neighbors

    def dom_wdeg_heuristic(self, assignment, csp):
        # print(self.weights)
        min_var = float('inf'), '', None
        # determine the unassigned variables

        for x in self.variables:
            if x in assignment.keys():
                continue
            # get the current domain of variable x
            dom = len(csp.choices(x))
            # get the list of unsigned neighbours
            wdeg = 1
            for y in self.neighbors[x]:
                if y in assignment.keys():
                    continue
                if (x, y) in self.weights.keys():
                    wdeg += self.weights[(x, y)]
                else:
                    wdeg += self.weights[(y, x)]
            if min_var[2] is None or dom/wdeg - min_var[0] < 10e-5:
                min_var = (dom/wdeg, str(x), x)

        return min_var[2]

    def FC_Solution(self):
        self.nassigns = 0
        self.weights = {pair: 1 for pair in self.ctr.keys()}
        # counter for constraints checks
        self.ctr_counter = 0

        start = time.time()
        try:
            with time_limit(200):
                ret = backtracking_search(self, order_domain_values=lcv,
                                  select_unassigned_variable=self.dom_wdeg_heuristic, inference=forward_checking)
        except Exception:
            print("Timeout Occured")
            ret = "Not Found"

        end = time.time()
        self.printStats(ret, "FC Backtracking", start, end)
        return ret

    def MAC_Solution(self):
        self.nassigns = 0
        self.weights = {pair: 1 for pair in self.ctr.keys()}
        # counter for constraints checks
        self.ctr_counter = 0

        start = time.time()
        try:
            with time_limit(200):
                ret = backtracking_search(self, order_domain_values=lcv,
                                  select_unassigned_variable=self.dom_wdeg_heuristic, inference=mac)
        except Exception:
            print("Timeout Occured")
            ret = "Not Found"

        end = time.time()
        self.printStats(ret, "MAC Backtracking", start, end)
        return ret

    def FC_CBJ_Solution(self):
        self.nassigns = 0
        self.weights = {pair: 1 for pair in self.ctr.keys()}
        # counter for constraints checks
        self.ctr_counter = 0

        # get starting time
        start = time.time()
        try:
            with time_limit(200):
                ret = backjumping_search(self, order_domain_values=lcv,
                                 select_unassigned_variable=self.dom_wdeg_heuristic, inference=forward_checking)
        except Exception:
            print("Timeout Occured")
            ret = "Not Found"
        end = time.time()
        self.printStats(ret, "FC-CBJ Backjumping", start, end)
        return ret

    def min_conflicts_Solution(self, max_steps=100000):
        self.nassigns = 0
        self.weights = {pair: 1 for pair in self.ctr.keys()}
        # counter for constraints checks
        self.ctr_counter = 0

        # get starting time
        start = time.time()
        try:
            with time_limit(200):
                ret = min_conflicts(self, max_steps)
        except Exception:
            print("Timeout Occured")
            ret = "Not Found"

        end = time.time()
        self.printStats(ret, "FC-CBJ Backjumping", start, end)
        return ret

    def printStats(self, assignment, title, start_timestamp, end_timestamp):
        print("++++", title, "Statistics ++++\nSolution:")
        self.display(assignment)
        dur = end_timestamp - start_timestamp
        print("Time: {:.3f} sec\nAssignments: {}\nConstraint Checks: {}\n".format(
            dur, self.nassigns, self.ctr_counter))


def backtracking_search(csp, select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values, inference=no_inference):
    """[Figure 6.5]"""

    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if csp.nassigns > csp.assignment_cutoff or csp.ctr_counter > csp.ctr_cutoff:
                        return "Not Finished"
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result == "Not Finished" or result is None or csp.goal_test(result)
    return result

def backjumping_search(csp, select_unassigned_variable=first_unassigned_variable,
                       order_domain_values=unordered_domain_values, inference=no_inference):
    """[Figure 6.5]"""
    csp.backjump_var = None  # the back jump variable pointer
    
    def FC(csp, var, value, assignment, removals, new_conflicts):
        csp.support_pruning()
        for B in csp.neighbors[var]:
            if B not in assignment:
                conflict = False
                for b in csp.curr_domains[B][:]:
                    if not csp.constraints(var, value, B, b):
                        csp.prune(B, b, removals)
                        # add the variable in the conflict set of its neighbour
                        conflict = True

                # domain wipeout
                if not csp.curr_domains[B]:
                    ######################
                    # increase the weights
                    key = (B, var)
                    if not (key in csp.weights.keys()):
                        key = (var, B)
                    csp.weights[key] += 1
                    #######################
                    return False
                if conflict:
                    csp.conflicts[B].add(var)
                    new_conflicts.append(B)
        return True

    def backjump(assignment):
        # print(assignment)
        if len(assignment) == len(csp.variables):
            return assignment, None
        # select variable based on given heuristic   
        var = select_unassigned_variable(assignment, csp)
        
        for val in order_domain_values(var, assignment, csp):
            new_conflicts = list()
            # assign the value only if the assignment is consistent
            if 0 == csp.nconflicts(var, val, assignment):
                csp.assign(var, val, assignment)
                removals = csp.suppose(var, val)

                # check if there is a conflict: if not then continue to the next assignment
                if FC(csp, var, val, assignment, removals, new_conflicts):
                    result, conflict_set = backjump(assignment)
                    # if the assignment is successful then return it
                    if result is not None:
                        return result, None 
                    
                    #if the assignment is not successfull then restore the conflict sets and the domains, unassign the variable and jumpback 
                    # till you find the first "backjump" call that the "var" variable is in the returned conflict set
                    # THIS WILL BE THE MOST RECENT CONFLICT (max_i{conflict_set of the failure variable})
                    if var not in conflict_set:
                        csp.restore(removals)
                        csp.unassign(var, assignment)
                        for v in new_conflicts:
                            csp.conflicts[v] -= set([var])
                        return None, conflict_set
                    else:
                        # in that case we have found the variable we want to back jump to.
                        # update the most recent conflict variable's set with the conflict set we propagate from the point of failure
                        csp.conflicts[var] = csp.conflicts[var].union(conflict_set).copy()
                
            #after a failure restore the domains and the conflict sets
            csp.restore(removals)
            csp.unassign(var, assignment)
            for v in new_conflicts:
                csp.conflicts[v] -= set([var])
        
        # cut off condition
        if csp.nassigns > csp.assignment_cutoff or csp.ctr_counter > csp.ctr_cutoff:
            return "Not Finished", None
        
        #return failure and the conflict set
        return None, set(csp.conflicts[var] - set([var]))

    result, cs = backjump({})
    assert result == "Not Finished" or result is None or csp.goal_test(result)
    return result


def AC3(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """[Figure 6.3]"""
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True, checks  # CSP is satisfiable


def revise(csp, Xi, Xj, removals, checks=0):
    """Return true if we remove a value."""
    revised = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        # if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
        conflict = True
        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False
            checks += 1
            if not conflict:
                break
        if conflict:
            csp.prune(Xi, x, removals)
            revised = True

    if not csp.curr_domains[Xi]:
        ########################
        key = (Xi, Xj)
        if not (key in csp.weights.keys()):
            key = (Xj, Xi)

        
        csp.weights[key] += 1
        #######################
    return revised, checks


def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    csp.support_pruning()
    for B in csp.neighbors[var]:
        if B not in assignment:
            conflict = False
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            # domain wipeout
            if not csp.curr_domains[B]:
                ######################
                key = (B, var)
                if not (key in csp.weights.keys()):
                    key = (var, B)

                csp.weights[key] += 1

                #######################
                return False
    return True


def mac(csp, var, value, assignment, removals, constraint_propagation=AC3):
    """Maintain arc consistency."""
    return constraint_propagation(csp, {(X, var) for X in csp.neighbors[var]}, removals)

def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None

if __name__ == "__main__":
    testcases = ["11", "2-f24", "2-f25", "3-f10", "3-f11",
                 "6-w2", "7-w1-f4", "7-w1-f5", "8-f10", "8-f11", "14-f27", "14-f28"]

    for instance in testcases:
        print("\n \nTestcase", instance)
        dom_path = "../rlfap/dom" + instance + ".txt"
        var_path = "../rlfap/var" + instance + ".txt"
        ctr_path = "../rlfap/ctr" + instance + ".txt"
        problem = RLFAP(dom_path, var_path, ctr_path)
        
        # problem.FC_Solution()
        # problem.MAC_Solution()
        problem.FC_CBJ_Solution()
        # problem.min_conflicts_Solution()
