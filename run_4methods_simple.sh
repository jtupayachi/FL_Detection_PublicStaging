#!/bin/bash
# 4-Method Sequential Orchestration - SIMPLIFIED (1 varying parameter per method)
# MOON, FedALA, StatAvg, DASHA running SEQUENTIALLY to avoid GPU OOM
# Each method completes all 12 experiments before the next method starts

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Parse command-line arguments
METHODS_TO_RUN=""
if [ $# -gt 0 ]; then
    METHODS_TO_RUN="$@"
else
    # Default: run all methods
    METHODS_TO_RUN="MOON FedALA StatAvg DASHA"
fi

# Validate method names
VALID_METHODS="MOON FedALA StatAvg DASHA"
for method in $METHODS_TO_RUN; do
    if ! echo "$VALID_METHODS" | grep -q "\<$method\>"; then
        echo -e "${RED}Error: Invalid method '$method'${NC}"
        echo -e "${YELLOW}Valid methods: MOON, FedALA, StatAvg, DASHA${NC}"
        echo -e "${YELLOW}Usage: $0 [MOON] [FedALA] [StatAvg] [DASHA]${NC}"
        echo -e "${YELLOW}Examples:${NC}"
        echo -e "  $0                    # Run all methods"
        echo -e "  $0 MOON               # Run only MOON"
        echo -e "  $0 MOON FedALA        # Run MOON and FedALA"
        echo -e "  $0 StatAvg DASHA      # Run StatAvg and DASHA"
        exit 1
    fi
done

EXPERIMENT_NAME="4METHODS_SIMPLE_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/home/jose/FL_IoT_Network/logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  4-Method Sequential FL Experiments (Fixed Params)${NC}"
echo -e "${BLUE}  Fixed parameters, varying only alpha (4 runs/method)${NC}"
echo -e "${BLUE}  Methods run ONE AT A TIME to avoid GPU OOM${NC}"
echo -e "${BLUE}  Running: ${METHODS_TO_RUN}${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"

# Check if containers are running and have GPU
echo -e "${YELLOW}Checking container status...${NC}"
if sudo docker ps | grep -q "fl_moon_server"; then
    # Check if GPU is available
    if sudo docker exec fl_moon_server python3 -c "import tensorflow as tf; exit(0 if tf.config.list_physical_devices('GPU') else 1)" 2>/dev/null; then
        echo -e "${GREEN}✓ Containers running with GPU - keeping them${NC}\n"
        SKIP_DATA=true
    else
        echo -e "${YELLOW}⚠ Containers running without GPU - restarting${NC}"
        sudo docker compose -f docker-compose-4methods.yml down
        sudo docker compose -f docker-compose-4methods.yml up -d
        sleep 10
        echo -e "${GREEN}✓ Containers restarted${NC}\n"
        SKIP_DATA=false
    fi
else
    echo -e "${YELLOW}Starting containers...${NC}"
    sudo docker compose -f docker-compose-4methods.yml up -d
    sleep 10
    echo -e "${GREEN}✓ Containers started${NC}\n"
    SKIP_DATA=false
fi

# Prepare data splits (run once in moon_server)
if [ "$SKIP_DATA" = false ]; then
    echo -e "${YELLOW}Preparing data splits...${NC}"
    alphas="0.001 0.01 0.1 1.0"

    for alpha in $alphas; do
        sudo docker exec fl_moon_server bash -c "
            cd /workspace && \
            python3 fl_testbed/version2/client/dirichelet_split.py \
            -data_X_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_inputs.pkl \
            -data_X_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_inputs.pkl \
            -data_y_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_out.pkl \
            -data_y_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_out.pkl \
            -cm 5 -alpha ${alpha} -beta 0.2 -motor 3 -type LSTM
        " 2>&1 | tee ${LOG_DIR}/datasplit_${alpha}.log
    done
    echo -e "${GREEN}✓ Data splits prepared${NC}\n"

    # Run independent training (once)
    echo -e "${YELLOW}Running independent training...${NC}"
    for client in 0 1 2 3 4; do
        sudo docker exec fl_moon_server bash -c "
            cd /workspace && \
            python3 fl_testbed/version2/client/independent.py -ml 2 -cn 15 -cm 15 -e 100 \
            -dfn 'M3_5_${client}_ddf_LSTM.pkl' \
            -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
            -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
            -ip 172.18.1.10
        " 2>&1 | tee ${LOG_DIR}/independent_client${client}.log &
    done
    wait
    echo -e "${GREEN}✓ Independent training done${NC}\n"
else
    echo -e "${BLUE}⊙ Skipping data preparation (already done)${NC}\n"
fi

# Clean up old result files
echo -e "${YELLOW}Cleaning up old result files...${NC}"
sudo docker exec fl_moon_server bash -c "cd /workspace && rm -f LSTM_*.txt"
sudo docker exec fl_fedala_server bash -c "cd /workspace && rm -f LSTM_*.txt"
sudo docker exec fl_statavg_server bash -c "cd /workspace && rm -f LSTM_*.txt"
sudo docker exec fl_dasha_server bash -c "cd /workspace && rm -f LSTM_*.txt"
echo -e "${GREEN}✓ Old files cleaned${NC}\n"

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  LAUNCHING ALL 4 METHODS SEQUENTIALLY${NC}"
echo -e "${BLUE}  Each method runs 4 experiments (4 alphas, fixed params)${NC}"
echo -e "${BLUE}  Total: 16 experiments${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"

# Function to run MOON experiment
run_moon_experiment() {
    local alpha=$1
    local tau_val=$2
    local port=$3
    
    echo -e "${YELLOW}MOON: alpha=$alpha tau=$tau_val port=$port${NC}"
    
    # Kill any lingering processes
    sudo docker exec fl_moon_server pkill -9 -f "federated_server_RUL_MOON" || true
    for cid in 0 1 2 3 4; do
        sudo docker exec fl_moon_client${cid} pkill -9 -f "federated_client_RUL_MOON" || true
    done
    sleep 2
    
    # Start server in background
    sudo docker exec -d fl_moon_server bash -c "cd /workspace && \
        python3 fl_testbed/version2/server/federated_server_RUL_MOON.py \
        -mu 1.0 -temperature 0.5 -tau ${tau_val} -slr 0.01 -cm 5 -e 1 --rounds 1000 \
        -ip 172.18.1.10:$port \
        -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
        -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
        -dfn 'M3_5_0_ddf_LSTM.pkl' \
        > LSTM_MOON_alpha_${alpha}_tau_${tau_val}.txt 2>&1"
    
    sleep 10  # Wait for server to start
    
    # Launch all 5 clients
    for client_id in 0 1 2 3 4; do
        sudo docker exec -d fl_moon_client${client_id} bash -c "cd /workspace && \
            python3 fl_testbed/version2/client/federated_client_RUL_MOON.py \
            -dfn 'M3_5_${client_id}_ddf_LSTM.pkl' \
            -cm 5 -e 1 -ip 172.18.1.10:$port -cn ${client_id} \
            -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
            -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
            > LSTM_MOON_client${client_id}_alpha_${alpha}_tau_${tau_val}.log 2>&1"
    done
    
    # Wait for FL to complete (monitor server process)
    while sudo docker exec fl_moon_server pgrep -f "federated_server_RUL_MOON" > /dev/null 2>&1; do
        sleep 10
    done
    
    echo -e "${GREEN}✓ MOON alpha=$alpha tau=$tau_val DONE${NC}"
}

# Function to run FedALA experiment
run_fedala_experiment() {
    local alpha=$1
    local tau_val=$2
    local port=$3
    
    echo -e "${YELLOW}FedALA: alpha=$alpha tau=$tau_val port=$port${NC}"
    
    # Kill any lingering processes
    sudo docker exec fl_fedala_server pkill -9 -f "federated_server_RUL_FedALA" || true
    for cid in 0 1 2 3 4; do
        sudo docker exec fl_fedala_client${cid} pkill -9 -f "federated_client_RUL_FedALA" || true
    done
    sleep 2
    
    sudo docker exec -d fl_fedala_server bash -c "cd /workspace && \
        python3 fl_testbed/version2/server/federated_server_RUL_FedALA.py \
        -tau ${tau_val} -slr 0.01 -cm 5 -e 1 --rounds 1000 \
        -ip 172.18.2.10:$port \
        -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
        -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
        -dfn 'M3_5_0_ddf_LSTM.pkl' \
        > LSTM_FedALA_alpha_${alpha}_tau_${tau_val}.txt 2>&1"
    
    sleep 10
    
    for client_id in 0 1 2 3 4; do
        sudo docker exec -d fl_fedala_client${client_id} bash -c "cd /workspace && \
            python3 fl_testbed/version2/client/federated_client_RUL_FedALA.py \
            -dfn 'M3_5_${client_id}_ddf_LSTM.pkl' \
            -cm 5 -e 1 -ip 172.18.2.10:$port -cn ${client_id} \
            -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
            -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
            > LSTM_FedALA_client${client_id}_alpha_${alpha}_tau_${tau_val}.log 2>&1"
    done
    
    while sudo docker exec fl_fedala_server pgrep -f "federated_server_RUL_FedALA" > /dev/null 2>&1; do
        sleep 10
    done
    
    echo -e "${GREEN}✓ FedALA alpha=$alpha tau=$tau_val DONE${NC}"
}

# Function to run StatAvg experiment
run_statavg_experiment() {
    local alpha=$1
    local slr_val=$2
    local port=$3
    
    echo -e "${YELLOW}StatAvg: alpha=$alpha slr=$slr_val port=$port${NC}"
    
    # Kill any lingering processes
    sudo docker exec fl_statavg_server pkill -9 -f "federated_server_RUL_StatAvg" || true
    for cid in 0 1 2 3 4; do
        sudo docker exec fl_statavg_client${cid} pkill -9 -f "federated_client_RUL_StatAvg" || true
    done
    sleep 2
    
    sudo docker exec -d fl_statavg_server bash -c "cd /workspace && \
        python3 fl_testbed/version2/server/federated_server_RUL_StatAvg.py \
        -tau 1.0 -slr ${slr_val} -cm 5 -e 1 --rounds 1000 \
        -ip 172.18.3.10:$port \
        -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
        -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
        -dfn 'M3_5_0_ddf_LSTM.pkl' \
        > LSTM_StatAvg_alpha_${alpha}_slr_${slr_val}.txt 2>&1"
    
    sleep 10
    
    for client_id in 0 1 2 3 4; do
        sudo docker exec -d fl_statavg_client${client_id} bash -c "cd /workspace && \
            python3 fl_testbed/version2/client/federated_client_RUL_StatAvg.py \
            -dfn 'M3_5_${client_id}_ddf_LSTM.pkl' \
            -cm 5 -e 1 -ip 172.18.3.10:$port -cn ${client_id} \
            -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
            -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
            > LSTM_StatAvg_client${client_id}_alpha_${alpha}_slr_${slr_val}.log 2>&1"
    done
    
    while sudo docker exec fl_statavg_server pgrep -f "federated_server_RUL_StatAvg" > /dev/null 2>&1; do
        sleep 10
    done
    
    echo -e "${GREEN}✓ StatAvg alpha=$alpha slr=$slr_val DONE${NC}"
}

# Function to run DASHA experiment
run_dasha_experiment() {
    local alpha=$1
    local slr_val=$2
    local port=$3
    
    echo -e "${YELLOW}DASHA: alpha=$alpha slr=$slr_val port=$port${NC}"
    
    # Kill any lingering processes
    sudo docker exec fl_dasha_server pkill -9 -f "federated_server_RUL_DASHA" || true
    for cid in 0 1 2 3 4; do
        sudo docker exec fl_dasha_client${cid} pkill -9 -f "federated_client_RUL_DASHA" || true
    done
    sleep 2
    
    sudo docker exec -d fl_dasha_server bash -c "cd /workspace && \
        python3 fl_testbed/version2/server/federated_server_RUL_DASHA.py \
        -tau 1.0 -slr ${slr_val} -cm 5 -e 1 --rounds 1000 \
        -ip 172.18.4.10:$port \
        -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
        -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
        -dfn 'M3_5_0_ddf_LSTM.pkl' \
        > LSTM_DASHA_alpha_${alpha}_slr_${slr_val}.txt 2>&1"
    
    sleep 10
    
    for client_id in 0 1 2 3 4; do
        sudo docker exec -d fl_dasha_client${client_id} bash -c "cd /workspace && \
            python3 fl_testbed/version2/client/federated_client_RUL_DASHA.py \
            -dfn 'M3_5_${client_id}_ddf_LSTM.pkl' \
            -cm 5 -e 1 -ip 172.18.4.10:$port -cn ${client_id} \
            -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
            -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
            > LSTM_DASHA_client${client_id}_alpha_${alpha}_slr_${slr_val}.log 2>&1"
    done
    
    while sudo docker exec fl_dasha_server pgrep -f "federated_server_RUL_DASHA" > /dev/null 2>&1; do
        sleep 10
    done
    
    echo -e "${GREEN}✓ DASHA alpha=$alpha slr=$slr_val DONE${NC}"
}

# Run experiments SEQUENTIALLY (one method at a time to avoid GPU OOM)
alphas="0.001 0.01 0.1 1.0"

# Fixed parameters for faster runs
FIXED_TAU=0.1
FIXED_SLR=0.01

PORT_MOON=8080
PORT_FEDALA=9000
PORT_STATAVG=10000
PORT_DASHA=11000

TOTAL_COMPLETED=0
TOTAL_EXPERIMENTS=$(echo "$METHODS_TO_RUN" | wc -w)
TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS * 4))

NUM_METHODS=$(echo "$METHODS_TO_RUN" | wc -w)

# Determine execution mode: parallel (2 methods) or sequential (1, 3, or 4 methods)
if [ $NUM_METHODS -eq 2 ]; then
    echo -e "${GREEN}Running 2 methods in PARALLEL (faster, uses more GPU memory)${NC}\n"
    
    # Convert methods to array
    methods_array=($METHODS_TO_RUN)
    method1=${methods_array[0]}
    method2=${methods_array[1]}
    
    # Run first method in background
    (
        if [ "$method1" = "MOON" ]; then
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
            echo -e "${BLUE}  METHOD: MOON (tau=$FIXED_TAU fixed)${NC}"
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
            for alpha in $alphas; do
                run_moon_experiment $alpha $FIXED_TAU $PORT_MOON
                PORT_MOON=$((PORT_MOON + 1))
            done
            echo -e "${GREEN}✓ MOON complete${NC}\n"
        elif [ "$method1" = "FedALA" ]; then
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
            echo -e "${BLUE}  METHOD: FedALA (tau=$FIXED_TAU fixed)${NC}"
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
            for alpha in $alphas; do
                run_fedala_experiment $alpha $FIXED_TAU $PORT_FEDALA
                PORT_FEDALA=$((PORT_FEDALA + 1))
            done
            echo -e "${GREEN}✓ FedALA complete${NC}\n"
        elif [ "$method1" = "StatAvg" ]; then
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
            echo -e "${BLUE}  METHOD: StatAvg (slr=$FIXED_SLR fixed)${NC}"
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
            for alpha in $alphas; do
                run_statavg_experiment $alpha $FIXED_SLR $PORT_STATAVG
                PORT_STATAVG=$((PORT_STATAVG + 1))
            done
            echo -e "${GREEN}✓ StatAvg complete${NC}\n"
        elif [ "$method1" = "DASHA" ]; then
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
            echo -e "${BLUE}  METHOD: DASHA (slr=$FIXED_SLR fixed)${NC}"
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
            for alpha in $alphas; do
                run_dasha_experiment $alpha $FIXED_SLR $PORT_DASHA
                PORT_DASHA=$((PORT_DASHA + 1))
            done
            echo -e "${GREEN}✓ DASHA complete${NC}\n"
        fi
    ) &
    
    # Run second method in background
    (
        if [ "$method2" = "MOON" ]; then
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
            echo -e "${BLUE}  METHOD: MOON (tau=$FIXED_TAU fixed)${NC}"
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
            for alpha in $alphas; do
                run_moon_experiment $alpha $FIXED_TAU $PORT_MOON
                PORT_MOON=$((PORT_MOON + 1))
            done
            echo -e "${GREEN}✓ MOON complete${NC}\n"
        elif [ "$method2" = "FedALA" ]; then
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
            echo -e "${BLUE}  METHOD: FedALA (tau=$FIXED_TAU fixed)${NC}"
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
            for alpha in $alphas; do
                run_fedala_experiment $alpha $FIXED_TAU $PORT_FEDALA
                PORT_FEDALA=$((PORT_FEDALA + 1))
            done
            echo -e "${GREEN}✓ FedALA complete${NC}\n"
        elif [ "$method2" = "StatAvg" ]; then
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
            echo -e "${BLUE}  METHOD: StatAvg (slr=$FIXED_SLR fixed)${NC}"
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
            for alpha in $alphas; do
                run_statavg_experiment $alpha $FIXED_SLR $PORT_STATAVG
                PORT_STATAVG=$((PORT_STATAVG + 1))
            done
            echo -e "${GREEN}✓ StatAvg complete${NC}\n"
        elif [ "$method2" = "DASHA" ]; then
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
            echo -e "${BLUE}  METHOD: DASHA (slr=$FIXED_SLR fixed)${NC}"
            echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
            for alpha in $alphas; do
                run_dasha_experiment $alpha $FIXED_SLR $PORT_DASHA
                PORT_DASHA=$((PORT_DASHA + 1))
            done
            echo -e "${GREEN}✓ DASHA complete${NC}\n"
        fi
    ) &
    
    # Wait for both methods to complete
    wait
    TOTAL_COMPLETED=$TOTAL_EXPERIMENTS
    
else
    # Sequential execution (1, 3, or 4 methods)
    echo -e "${YELLOW}Running methods SEQUENTIALLY (safer, less GPU memory)${NC}\n"
    
    # Method 1: MOON (4 experiments)
    if echo "$METHODS_TO_RUN" | grep -q "\<MOON\>"; then
        echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}  METHOD: MOON (tau=$FIXED_TAU fixed)${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
        for alpha in $alphas; do
            run_moon_experiment $alpha $FIXED_TAU $PORT_MOON
            PORT_MOON=$((PORT_MOON + 1))
            TOTAL_COMPLETED=$((TOTAL_COMPLETED + 1))
        done
        echo -e "${GREEN}✓ MOON complete ($TOTAL_COMPLETED/$TOTAL_EXPERIMENTS)${NC}\n"
        sleep 5
    fi

    # Method 2: FedALA (4 experiments)
    if echo "$METHODS_TO_RUN" | grep -q "\<FedALA\>"; then
        echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}  METHOD: FedALA (tau=$FIXED_TAU fixed)${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
        for alpha in $alphas; do
            run_fedala_experiment $alpha $FIXED_TAU $PORT_FEDALA
            PORT_FEDALA=$((PORT_FEDALA + 1))
            TOTAL_COMPLETED=$((TOTAL_COMPLETED + 1))
        done
        echo -e "${GREEN}✓ FedALA complete ($TOTAL_COMPLETED/$TOTAL_EXPERIMENTS)${NC}\n"
        sleep 5
    fi

    # Method 3: StatAvg (4 experiments)
    if echo "$METHODS_TO_RUN" | grep -q "\<StatAvg\>"; then
        echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}  METHOD: StatAvg (slr=$FIXED_SLR fixed)${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
        for alpha in $alphas; do
            run_statavg_experiment $alpha $FIXED_SLR $PORT_STATAVG
            PORT_STATAVG=$((PORT_STATAVG + 1))
            TOTAL_COMPLETED=$((TOTAL_COMPLETED + 1))
        done
        echo -e "${GREEN}✓ StatAvg complete ($TOTAL_COMPLETED/$TOTAL_EXPERIMENTS)${NC}\n"
        sleep 5
    fi

    # Method 4: DASHA (4 experiments)
    if echo "$METHODS_TO_RUN" | grep -q "\<DASHA\>"; then
        echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
        echo -e "${BLUE}  METHOD: DASHA (slr=$FIXED_SLR fixed)${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"
        for alpha in $alphas; do
            run_dasha_experiment $alpha $FIXED_SLR $PORT_DASHA
            PORT_DASHA=$((PORT_DASHA + 1))
            TOTAL_COMPLETED=$((TOTAL_COMPLETED + 1))
        done
        echo -e "${GREEN}✓ DASHA complete ($TOTAL_COMPLETED/$TOTAL_EXPERIMENTS)${NC}\n"
    fi
fi

echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ALL EXPERIMENTS COMPLETED! ($TOTAL_COMPLETED total)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}\n"

# Collect results
echo -e "${YELLOW}Collecting results...${NC}"
if echo "$METHODS_TO_RUN" | grep -q "\<MOON\>"; then
    sudo docker cp fl_moon_server:/workspace/LSTM_MOON_*.txt ./ 2>/dev/null || echo "No MOON results found"
fi
if echo "$METHODS_TO_RUN" | grep -q "\<FedALA\>"; then
    sudo docker cp fl_fedala_server:/workspace/LSTM_FedALA_*.txt ./ 2>/dev/null || echo "No FedALA results found"
fi
if echo "$METHODS_TO_RUN" | grep -q "\<StatAvg\>"; then
    sudo docker cp fl_statavg_server:/workspace/LSTM_StatAvg_*.txt ./ 2>/dev/null || echo "No StatAvg results found"
fi
if echo "$METHODS_TO_RUN" | grep -q "\<DASHA\>"; then
    sudo docker cp fl_dasha_server:/workspace/LSTM_DASHA_*.txt ./ 2>/dev/null || echo "No DASHA results found"
fi

echo -e "${GREEN}✓ Results copied to host${NC}\n"

ls -lh LSTM_*.txt | wc -l
echo "experiment result files generated"
