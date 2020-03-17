#!/usr/bin/env bash

OUTPUT_FOLDER="output/$(date +%Y_%m_%d_%H_%M_%S)/"
mkdir -p "${OUTPUT_FOLDER}"

declare -a CHANGE_TYPE=(1 2 3)
declare -a FUTURE_HORIZON=(2 6 10)
declare -a ALGORITHM=(1)

MAX_RUNS=5
SEED=12
MAX_CHANGES=23
CHANGE_FREQUENCY=2500
POPULATION_SIZE=50
OUTPUT_FILE=""
JAR_FILE="root-ams-0.1-all.jar"

ARG_LIST=()
RESULT_ID=0
for i in "${CHANGE_TYPE[@]}" ; do
    for j in "${FUTURE_HORIZON[@]}" ; do
      for k in "${ALGORITHM[@]}" ; do
        for RUN_ID in {1..${MAX_RUNS}} ; do
          ((RESULT_ID+=1))
          OUTPUT_FILE="${OUTPUT_FOLDER}RESULT_${RESULT_ID}.CSV"
          ARG_LIST+=("-jar ${JAR_FILE} ${RUN_ID} ${SEED} ${MAX_CHANGES} ${CHANGE_FREQUENCY} ${i} ${j} ${k} ${POPULATION_SIZE} ${OUTPUT_FILE}")
          done;
        done;
    done;
done


printf "%s\n" "${ARG_LIST[@]}" | xargs -t -P 3 -n 11 java

echo "Experiment were done. Results were saved in folder ${OUTPUT_FOLDER}"

echo "Making the final arragements"

cd "$OUTPUT_FOLDER"

(head -1 RESULT_1.CSV ; tail -n +2 -q RESULT_* ) > ALL_RESULTS.CSV