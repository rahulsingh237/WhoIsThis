import csv
import Image_loader as im

def register(fname,lname,ID,age,password):
    file = open('data.csv','a',newline='')
    row = (fname,lname,ID,age,password)
    writer = csv.writer(file)
    writer.writerow(row)
    file.close()
    images(fname)


def images(name):
    filename=name
    print('Initiating Image loader')
    print('Please look into the camera for accurate results')
    print('Starting in 3..2..1...')
    im.main(filename)

def newUser():
    fname = input("\n\nEnter your first name: ")
    lname = input('\n\nEnter your last name: ')
    a = int(input("\n\nEnter age: "))
    i = int(input("\n\nEnter id: "))
    p = input("\n\nEnter your password: ")
    register(fname,lname,i,a,p)




def existingUser():
    i = input("\n\nEnter ID: ")
    iterator = 3
    csv_file = csv.reader(open('data.csv', "r"), delimiter=",")
    for row in csv_file:
        if  row[2] == i:
            while(iterator > 0):
                p = input("\n\nEnter password: ")
                if row[4]==p:
                    print("\n\n\t\tWelcome to the network",row[0],row[1])
                    break
                else:
                    print("\n\nWrong password !!")
                    iterator = iterator-1
                    print("\n\nYou have ",iterator," attempts left !!")
            if iterator == 0: 
                print("\n\n\t\tReturning to Welcome Page")
                welcomePage()
            break
    else:
        print("\n\nPlease enter a valid id ")
        print("\n\n\t\tReturning to the Welcome Page...")
        welcomePage()

def welcomePage():
    print("\t\t\n\nWelcome")
    c = input("\n\nPress S for SIGN UP,  L for LOG IN and X to exit  ")
    if(c=='s' or c=='S'):
        newUser()
    if(c=='l' or c=='L'):
        existingUser()
    if(c=='x' or c=='X'):
        print("\n\nThank you")

welcomePage()