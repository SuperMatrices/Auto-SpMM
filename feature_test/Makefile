CXX = g++

CXXFLAGS = -std=c++14 -O3 -fopenmp
DEBUGFLAG = -g

feature_collect: feature_collect.o feature_generation.o mtxio.o base.o
	$(CXX) $(CXXFLAGS) $^ -o $@

feature_collect.o: feature_collect.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

feature_generation.o: ./feature_generation.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

base.o: ./base.cc	
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

mtxio.o: ./mtxio.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

clean:
	rm -f *.o feature_collect