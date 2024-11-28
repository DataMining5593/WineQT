




with open("winequality-all.csv", "w") as rd:
    with open("winequality-red.csv", "r") as rouge:
        for l in rouge:
            if(l[0] == "\""):
                rd.write("\"color\";")
                rd.write(l)
            else:
                rd.write("1;")
                rd.write(l)
    with open("winequality-white.csv", "r") as w:
        for l in w:
            if(l[0] != "\""):
                rd.write("0;")
                rd.write(l)