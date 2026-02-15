EIGEN_PATH = /usr/include/eigen3

CXX = g++

CXXFLAGS = -O3 -std=c++11 -I$(EIGEN_PATH)

TARGET = quantum_gen

SRC = data_gen.cpp funzioni.h

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)