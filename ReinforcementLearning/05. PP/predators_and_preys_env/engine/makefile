game.so: game.o
	gcc -shared -o game.so game.o
	
game.o:
	gcc -c game.c

clean:
	rm -f game.so game.o