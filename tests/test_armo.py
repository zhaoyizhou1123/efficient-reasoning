from rm import ArmoRM

if __name__ == "__main__":
    rm = ArmoRM()
    score = rm.evaluate(prompt="Bro, what is 2+2", response="Bro, it's 10")
    print(score)
