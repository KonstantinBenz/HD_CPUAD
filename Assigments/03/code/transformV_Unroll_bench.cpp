#include <numeric>
#include <cmath>
#include "benchmarker.hpp"
#include "sol_transform_LoopUnrolling_view.hpp"

int main(){
	using transform = transform_LoopUnrolling_view<ClockRecorder>;
	int Nl = 30;
	int Nh = 31;
    auto timings = 100;
    ClockRecorder rec(timings+1);       // recordings
    std::vector<long int> dur(timings); // durations in nano 
    transform_LoopUnrolling_view<ClockRecorder> test(rec);
	
	std::cout << "name,size,avg throughput,avg time,stdev time,cv\n";
	//benchmarker(&test, &transform::benchTransformOmpSimd, "benchTransformOmpSimd", Nl, Nh, dur, timings,rec);
	//benchmarker(&test, &transform::benchTransformDirectiveUnroll, "benchTransformDirectiveUnroll", Nl, Nh, dur, timings,rec);
	//benchmarker(&test, &transform::benchTransformDirectiveUnrollFactor64, "benchTransformDirectiveUnrollFactor64", Nl, Nh, dur, timings,rec);
	// benchmarker(&test, &transform::benchTransformUnrollManual, "benchTransformUnrollManual", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchTransformUnrollLoopPeeling, "benchTransformUnrollLoopPeeling", Nl, Nh, dur, timings,rec);
	benchmarker(&test, &transform::benchTransformUnrollLoopPeelingDirective, "benchTransformUnrollLoopPeelingDirective", Nl, Nh, dur, timings,rec);
	

    std::cout << test.get_log();

	return 0;
}
