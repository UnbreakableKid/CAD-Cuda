//
// Created by Hervé Paulino on 27/09/2020.
//

#include <gtest/gtest.h>
#include <fstream>
#include <algorithm>

#include <nbody/cuda_nbody_all_pairs.h>

/**
 * Compares the result produced and stored in stream result_stream with the log in file logfilename
 *
 * @param result_stream
 * @param filename
 */
void compare_results(std::stringstream& result_stream, std::string& logfilename) {
    std::string bufferResult;
    std::string bufferExpected;
    std::ifstream ins(logfilename);
    ASSERT_TRUE(ins.is_open());

    while (getline (result_stream, bufferResult)) {
        getline (ins, bufferExpected);
        std::string expected;
        expected.reserve(bufferExpected.size());
        std::copy_if(bufferExpected.begin(), bufferExpected.end(), std::back_inserter(expected),
                     [] (char c) { return c != '\r'; });

        EXPECT_STREQ(bufferResult.c_str(), expected.c_str());
    }

    ins.close();
}


TEST(NBody, CUDA_All_Pairs_P1000_T10_U0_T4) {

    auto nparticles = 1000;
    auto T_FINAL = 10.0;
    auto universe = cadlabs::universe_t::ORIGINAL;
    auto number_of_threads = 4;
    std::string original_result_log = "p1000_t10_u0.log";

    cadlabs::cuda_nbody_all_pairs nbody(nparticles, T_FINAL, number_of_threads, universe);
    nbody.run_simulation();

    std::stringstream ss;
    nbody.print_all_particles(ss);
    compare_results(ss, original_result_log);
}
