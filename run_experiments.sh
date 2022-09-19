if (($# < 1))
then
    echo "Missing required agent name argument"
    echo "Usage run_experiment.sh AGENT_NAME [NUM_ITER [EPISODE_TIME_LIMIT]]"
    exit 1
fi
a=${1}
n=${2:-10}
t=${3:-7200}
for (( i=1; i<=n; i++ ))
do
    python3 wh_create_model.py -a "$a.json" -m "models/$a"
    python3 wh_learn_enc.py -s $a -t $t -m "models/$a"
done