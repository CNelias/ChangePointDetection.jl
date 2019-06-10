using Test
using EditDistance

@test Levenshtein("sunday","saturday") == 3
@test DamerauLevenshtein("sunday","saturday") == 3
