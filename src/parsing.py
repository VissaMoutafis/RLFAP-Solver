def simple_parsing(filename):
    with open(filename) as f:
        lines = f.readlines()
        t = [None, None, None]
        c = 0
        for line in lines:
            if c == 3:
                print("({})".format(','.join([str(i if i < 1 else int(i)) for i in t])))
                c = 0
            if "Testcase" in line:
                print(line[len("Testcase:"):-1], end=': ')
            if "Time" in line and "Timeout" not in line:
                try:
                    attr = float(line[len("Time:"):-5])
                except Exception:
                    attr = '-'
                c+=1
                t[0] = attr
            if "Assignments" in line:
                attr = int(line[len("Assignments:"):])
                c += 1
                t[1] = attr
            if "Constraint Checks" in line:
                attr = int(line[len("Constraint Checks:"):])
                c += 1
                t[2] = attr
        print("({})".format(','.join([str(i if i < 1 else int(i)) for i in t])))


def complex_parsing(filenames):
    times = {}
    assignments = {}
    constraint_checks = {}
    
    def add_record(testcase, t, a, c):
        if testcase in times:
            times[testcase].append(t)
        else:
            times[testcase] = [t]
        
        if testcase in assignments:
            assignments[testcase].append(a)
        else:
            assignments[testcase] = [a]
    
        if testcase in constraint_checks:
            constraint_checks[testcase].append(c)
        else:
            constraint_checks[testcase] = [c]

    for filename in filenames:
        with open(filename) as f:
            testcase_name = None
            lines = f.readlines()
            t = [None, None, None]
            c = 0
            for line in lines:
                if c == 3:
                    # print("({})".format(','.join([str(i) for i in t])))
                    add_record(testcase_name, t[0], t[1], t[2])
                    c = 0
                if "Testcase" in line:
                    # print(line[len("Testcase:"):-1], end=': ')
                    testcase_name = line[len("Testcase:"):-1]
                if "Time" in line and "Timeout" not in line:
                    try:
                        attr = float(line[len("Time:"):-5])
                    except Exception:
                        attr = '-'
                    c += 1
                    t[0] = attr
                if "Assignments" in line:
                    attr = int(line[len("Assignments:"):])
                    c += 1
                    t[1] = attr
                if "Constraint Checks" in line:
                    attr = int(line[len("Constraint Checks:"):])
                    c += 1
                    t[2] = attr
            # print("({})".format(','.join([str(i) for i in t])))
            add_record(testcase_name, t[0], t[1], t[2])
    
    for test in times.keys():
        print(test, round(sum(times[test])/len(times[test]), 2),
              round(sum(assignments[test])/len(assignments[test]), 2), round(sum(constraint_checks[test])/len(constraint_checks[test]), 2), sep=" & ")

if __name__ == "__main__":
    filenames = ["FC Results.txt", "MAC Results.txt", "FC-CBJ Results.txt"]
    min_confs_files = ["minconf_{}.txt".format(s) for s in ["10", "100", "1K", "10K", "100K"]]

    # complex_parsing(min_confs_files)
    for n in filenames:
        print(n)
        simple_parsing(n)
        print("\n \n \n")