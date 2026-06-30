main: main.cu
	nvcc -O3 main.cu -o main
	nvcc -O3 main2.cu -o main2
	nvcc -O3 -arch=sm_86 main_MMP.cu -o main_MMP
	nvcc -O3 -arch=sm_86 main_MMP_share.cu -o main_MMP_share
	nvcc -O3 -arch=sm_86 main_MMP_pipeline.cu -o main_MMP_pipeline


run:
	./main
	./main2
	./main_MMP
	./main_MMP_share
	./main_MMP_pipeline
	
clean:
	rm -f main
	rm -f main2
	rm -f main_MMP
	rm -f main_MMP_share
	rm -f main_MMP_pipeline