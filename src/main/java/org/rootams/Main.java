package org.rootams;

import com.workday.insights.timeseries.arima.Arima;
import com.workday.insights.timeseries.arima.struct.ArimaParams;
import com.workday.insights.timeseries.arima.struct.ForecastResult;
import smile.base.rbf.RBF;
import smile.clustering.KMeans;
import smile.math.distance.EuclideanDistance;
import smile.math.rbf.GaussianRadialBasis;
import smile.regression.RBFNetwork;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.*;


public class Main {

    static int MAX_RUNS;
    static int SEED;
    static int MAX_CHANGES;
    static int CHANGE_FREQUENCY;
    static int CHANGE_TYPE;
    static int FUTURE_HORIZON;
    static int ALGORITHM_ID;
    static int POPULATION_SIZE;
    static String OUTPUT_FILE;

    static int RUN_ID;
    static double[] INITIAL_RECORD;

    public static void main(String[] args) {
        if (args == null || args.length == 0) {
            System.out.println("Arguments required...");
            System.exit(0);
        }

        try {

            MAX_RUNS = Integer.parseInt(args[0]);
            SEED = Integer.parseInt(args[1]);
            MAX_CHANGES = Integer.parseInt(args[2]);
            CHANGE_FREQUENCY = Integer.parseInt(args[3]);
            CHANGE_TYPE = Integer.parseInt(args[4]);
            FUTURE_HORIZON = Integer.parseInt(args[5]);
            ALGORITHM_ID = Integer.parseInt(args[6]);
            POPULATION_SIZE = Integer.parseInt(args[7]);
            OUTPUT_FILE = args[8];

            printHeadOfOutputFile();

            if (ALGORITHM_ID == 1) {
                runPSOPerfectEvaluation();
            } else if (ALGORITHM_ID == 2) {
                runJinApproach();
            }
        } catch (IllegalArgumentException exception) {

        }
    }

    interface RobustnessEvaluator {

        double evaluate(RMPBI problem, double[] x);
    }

    static class PerfectEvaluator implements RobustnessEvaluator {

        @Override
        public double evaluate(RMPBI problem, double[] x) {
            return problem.trueEval(x);
        }
    }

    static class PresentEvaluator implements RobustnessEvaluator {

        @Override
        public double evaluate(RMPBI problem, double[] x) {
            return problem.eval(x);
        }
    }

    static class DataBaseEnvironment{

        public double[][] dataBaseX;
        public double[] dataBaseY;

        public void append(double[][] dataBaseX, double[] dataBaseY){
            if(this.dataBaseY == null){
                int sampleSize = dataBaseX.length;
                int dimension = dataBaseX[0].length;
                this.dataBaseX = new double[sampleSize][dimension];
                for (int i = 0; i < dataBaseX.length; i++) {
                    dataBaseX[i] = Arrays.copyOf(dataBaseX[i], dimension);
                }
                this.dataBaseY = Arrays.copyOf(dataBaseY, sampleSize);
            }
            else{
                this.dataBaseX = Utils.appendMatrixToMatrix(dataBaseX, this.dataBaseX);
                this.dataBaseY = Utils.appendVectorToVector(dataBaseY, this.dataBaseY);
            }

        }
    }

    static class Swarm{

        public final double C = 1.496;
        public final double W = .729;

        public double[][] x;
        public double[] f ;
        public double[][] px;
        public double[] pf;
        public double[][] v;

        public double gf;
        public double[] gx;


        public Swarm(int swarmSize, int dimension){

            this.x = new double[swarmSize][dimension];
            this.f = new double[swarmSize];
            this.v = new double[swarmSize][dimension];
            this.px = new double[swarmSize][dimension];
            this.pf = new double[swarmSize];
            this.gx = new double[dimension];
            this.gf = Double.NEGATIVE_INFINITY;
        }
    }

    public static void runJinApproach() {

        for (RUN_ID = 0; RUN_ID < MAX_RUNS ; RUN_ID++) {

            int algorithmSeed = RUN_ID+1;
            Random rand = new Random(algorithmSeed);
            RMPBI problem = createProblem();

            Swarm swarm = new Swarm(POPULATION_SIZE, problem.dimension);

            int currentIteration = 0;
            int currentChange = 0;
            int maxIterations = CHANGE_FREQUENCY/POPULATION_SIZE;

            PresentEvaluator evaluator = new PresentEvaluator();
            JinEvaluator jinEvaluator = new JinEvaluator();

            initializeSwarm(swarm, problem, evaluator, rand);
            printPerformance(RUN_ID, currentChange, currentIteration, problem.trueEval(swarm.gx));
            jinEvaluator.createNewEnvironment(problem);
            jinEvaluator.saveEnvironmentData(swarm.x, swarm.f);

            int iterInit = 1;
            for (currentChange = 0; currentChange < problem.learningPeriod; currentChange++) {
                for (int iter = iterInit;  iter < maxIterations; iter++) {
                    //currentIteration++;
                    evolveSwarm(swarm, problem, evaluator, rand);
                    printPerformance(RUN_ID, currentChange, iter, problem.trueEval(swarm.gx));
                    jinEvaluator.saveEnvironmentData(swarm.x, swarm.f);
                }
                problem.change();
                updateSwarm(swarm, problem, evaluator, rand);
                printPerformance(RUN_ID, currentChange, 0, problem.trueEval(swarm.gx));
                reinitializeSwarm(swarm, problem, evaluator, rand);
                printPerformance(RUN_ID, currentChange, 1, problem.trueEval(swarm.gx));
                jinEvaluator.createNewEnvironment(problem);
                jinEvaluator.saveEnvironmentData(swarm.x, swarm.f);
                iterInit = 2;
            }

            for (currentChange=problem.learningPeriod; currentChange < MAX_CHANGES; currentChange++) {

                for (int iter = iterInit;  iter < maxIterations; iter++) {
                    //currentIteration++;
                    evolveSwarm(swarm, problem, jinEvaluator, rand);
                    printPerformance(RUN_ID, currentChange, iter, problem.trueEval(swarm.gx));
                    jinEvaluator.saveEnvironmentData(swarm.x, swarm.f);
                }
                problem.change();
                updateSwarm(swarm, problem, jinEvaluator, rand);
                printPerformance(RUN_ID, currentChange, 0, problem.trueEval(swarm.gx));
                reinitializeSwarm(swarm, problem, jinEvaluator, rand);
                printPerformance(RUN_ID, currentChange, 1, problem.trueEval(swarm.gx));
                jinEvaluator.createNewEnvironment(problem);
                jinEvaluator.saveEnvironmentData(swarm.x, swarm.f);
                iterInit = 2;
            }
        }

    }

    public static void runPSOPerfectEvaluation() {

        for (RUN_ID = 0; RUN_ID < MAX_RUNS ; RUN_ID++) {

            int algorithmSeed = RUN_ID+1;
            Random rand = new Random(algorithmSeed);
            RMPBI problem = createProblem();

            Swarm swarm = new Swarm(POPULATION_SIZE, problem.dimension);


            int currentIteration = 0;
            int currentChange = 0;
            int maxIterations = CHANGE_FREQUENCY/POPULATION_SIZE;

            //PresentEvaluator presentEvaluator = new PresentEvaluator();
            PerfectEvaluator perfectEvaluator = new PerfectEvaluator();

            initializeSwarm(swarm, problem, perfectEvaluator, rand);
            printPerformance(RUN_ID, currentChange, currentIteration, problem.trueEval(swarm.gx));

            int iterInit = 1;
            for (currentChange = 0; currentChange < MAX_CHANGES; currentChange++) {
                for (int iter = iterInit;  iter < maxIterations; iter++) {
                    //currentIteration++;
                    evolveSwarm(swarm, problem, perfectEvaluator, rand);
                    printPerformance(RUN_ID, currentChange, iter, problem.trueEval(swarm.gx));
                }
                problem.change();
                updateSwarm(swarm, problem, perfectEvaluator, rand);
                printPerformance(RUN_ID, currentChange, 0, problem.trueEval(swarm.gx));
                reinitializeSwarm(swarm, problem, perfectEvaluator, rand);
                printPerformance(RUN_ID, currentChange, 1, problem.trueEval(swarm.gx));
                iterInit = 2;
            }
        }
    }

    static class JinEvaluator implements RobustnessEvaluator {

        int AR_ORDER = 4;
        int K = 20;
        LinkedList<DataBaseEnvironment> pastEnvironments = new LinkedList<>();
        DataBaseEnvironment currentEnvironment;

        public void createNewEnvironment(RMPBI problem) {
            this.pastEnvironments.addLast(new DataBaseEnvironment());
            this.currentEnvironment = this.pastEnvironments.getLast();
            if (this.pastEnvironments.size() > problem.learningPeriod) {
                this.pastEnvironments.removeFirst();
            }
        }

        class SortableStructure implements Comparable<SortableStructure> {

            double distance;
            int originalIndex;

            @Override
            public int compareTo(SortableStructure o) {
                if (distance > o.distance)
                    return 1;
                if (distance < o.distance)
                    return -1;
                return 0;
            }
        }


        @Override
        public double evaluate(RMPBI problem, double[] x) {

            double[] pastY = evaluateInThePast(x);

            double presentY = problem.eval(x);

            double[] timeSeries = Utils.appendValueToVector(presentY, pastY);

            double[] futureY = evaluateInTheFuture(timeSeries, problem);

            return problem.assistedEval(presentY, futureY);
        }

        public double[] evaluateInThePast(double[] x) {

            double[] result = new double[pastEnvironments.size()];
            for (int i = 0; i < pastEnvironments.size(); i++) {
                result[i] = evalPastEnvironment(pastEnvironments.get(i), x);
            }
            return result;
        }

        public double[] evaluateInTheFuture(double[] timeSeries, RMPBI problem) {

            ArimaParams arp = new ArimaParams(AR_ORDER, 0, 0, 0, 0, 0, 0);

            ForecastResult forecastResult = Arima.forecast_arima(timeSeries, problem.timeWindows - 1, arp);

            return forecastResult.getForecast();
        }

        private double evalPastEnvironment(DataBaseEnvironment environment, double[] x) {

            EuclideanDistance euclideanDistance = new EuclideanDistance();

            ArrayList<SortableStructure> listDistances = new ArrayList<>();

            for (int i = 0; i < environment.dataBaseX.length; i++) {

                double distance = euclideanDistance.d(environment.dataBaseX[i], x);

                if (distance == 0) {
                    return environment.dataBaseY[i];
                }
                SortableStructure sortable = new SortableStructure();
                sortable.originalIndex = i;
                sortable.distance = distance;
                listDistances.add(sortable);
            }

            Collections.sort(listDistances);

            // Fit a RBF Network with K nearest points to x

            int trainingSize = 2 * K;

            double[][] trainingX = new double[trainingSize][x.length];
            double[] trainingY = new double[trainingSize];

            for (int i = 0; i < trainingSize; i++) {
                int index = listDistances.get(i).originalIndex;
                System.arraycopy(environment.dataBaseX[index], 0, trainingX[i], 0, x.length);
                trainingY[i] = environment.dataBaseY[index];
            }

            KMeans kMeans = KMeans.fit(trainingX, this.K);

            RBF<double[]>[] rbfs = RBF.of(kMeans.centroids, new GaussianRadialBasis(), euclideanDistance);
            RBFNetwork<double[]> rbfNetwork = RBFNetwork.fit(trainingX, trainingY, rbfs);

            return rbfNetwork.predict(x);
        }


        public void saveEnvironmentData(double[][] x, double[] y) {

            this.currentEnvironment.append(x, y);
        }


    }

    public static RMPBI createProblem() {

        SEED += 123;
        RMPBI problem = new RMPBI();
        problem.seed = SEED;
        problem.computationalBudget = CHANGE_FREQUENCY;
        problem.timeWindows = FUTURE_HORIZON;
        problem.changeType = CHANGE_TYPE;
        problem.init();
        return problem;

    }

    public static void initializeSwarm(Swarm swarm, RMPBI problem, RobustnessEvaluator evaluator, Random rand){

            int SWARM_SIZE = swarm.x.length;
            int PROBLEM_DIMENSION = problem.dimension;
            double LOWER_BOUND = problem.minCoord;
            double UPPER_BOUND = problem.maxCoord;

            for (int i = 0; i < SWARM_SIZE; i++) {
                for (int j = 0; j < PROBLEM_DIMENSION; j++) {
                    swarm.x[i][j] = LOWER_BOUND + (UPPER_BOUND-LOWER_BOUND)*rand.nextDouble();
                    swarm.px[i][j] = swarm.x[i][j];
                    swarm.v[i][j] = LOWER_BOUND + (UPPER_BOUND-LOWER_BOUND)*rand.nextDouble();
                    swarm.v[i][j] = 0.5 * (swarm.x[i][j] - swarm.v[i][j]);
                }
                swarm.f[i] = evaluator.evaluate(problem, swarm.x[i]);
                swarm.pf[i] = swarm.f[i];
                if (swarm.pf[i] > swarm.gf) {
                    swarm.gx = Arrays.copyOf(swarm.px[i], PROBLEM_DIMENSION);
                    swarm.gf = swarm.pf[i];
                }
            }

    }

    public static void evolveSwarm(Swarm swarm, RMPBI problem, RobustnessEvaluator evaluator, Random rand){

        int SWARM_SIZE = swarm.x.length;
        int PROBLEM_DIMENSION = problem.dimension;
        double LOWER_BOUND = problem.minCoord;
        double UPPER_BOUND = problem.maxCoord;


        for (int i = 0; i < SWARM_SIZE; i++) {
            for (int d = 0; d < PROBLEM_DIMENSION; d++) {

                double social = rand.nextDouble() * swarm.C * (swarm.gx[d] - swarm.x[i][d]);

                double cognition = rand.nextDouble() * swarm.C * (swarm.px[i][d] - swarm.x[i][d]);

                double v = swarm.W * swarm.v[i][d] + social + cognition;

                double x = swarm.x[i][d] + v;

                x = Utils.clampValue(x, LOWER_BOUND, UPPER_BOUND);//TODO: fix this dependency

                if (x == LOWER_BOUND || x == UPPER_BOUND) {
                    v = 0.;
                }

                swarm.x[i][d] = x;
                swarm.v[i][d] = v;
            }
            swarm.f[i] = evaluator.evaluate(problem, swarm.x[i]);

            if(swarm.f[i] > swarm.pf[i]){
                swarm.px[i] = Arrays.copyOf(swarm.x[i], PROBLEM_DIMENSION);
                swarm.pf[i] = swarm.f[i];

                if(swarm.pf[i] > swarm.gf){
                    swarm.gx = Arrays.copyOf(swarm.px[i], PROBLEM_DIMENSION);
                    swarm.gf = swarm.pf[i];
                }

            }
        }

    }

    public static void updateSwarm(Swarm swarm, RMPBI problem, RobustnessEvaluator evaluator, Random rand){

        int SWARM_SIZE = swarm.x.length;
        int PROBLEM_DIMENSION = problem.dimension;

        swarm.gf = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < SWARM_SIZE; i++) {

            swarm.pf[i] = evaluator.evaluate(problem, swarm.px[i]);

            if(swarm.pf[i] > swarm.gf){
                swarm.gx = Arrays.copyOf(swarm.px[i], PROBLEM_DIMENSION);
                swarm.gf = swarm.pf[i];
            }
        }
    }

    public static void reinitializeSwarm(Swarm swarm, RMPBI problem, RobustnessEvaluator evaluator, Random rand){

        int SWARM_SIZE = swarm.x.length;
        int PROBLEM_DIMENSION = problem.dimension;
        double LOWER_BOUND = problem.minCoord;
        double UPPER_BOUND = problem.maxCoord;

        for (int i = 0; i < SWARM_SIZE; i++) {
            for (int j = 0; j < PROBLEM_DIMENSION; j++) {
                swarm.x[i][j] = LOWER_BOUND + (UPPER_BOUND-LOWER_BOUND)*rand.nextDouble();
                swarm.v[i][j] = LOWER_BOUND + (UPPER_BOUND-LOWER_BOUND)*rand.nextDouble();
                swarm.v[i][j] = 0.5*(swarm.x[i][j] - swarm.v[i][j]);
            }
            swarm.f[i] = evaluator.evaluate(problem, swarm.x[i]);
            if(swarm.f[i] > swarm.pf[i]){
                swarm.px[i] = Arrays.copyOf(swarm.x[i], PROBLEM_DIMENSION);
                swarm.pf[i] = swarm.f[i];

                if(swarm.pf[i] > swarm.gf){
                    swarm.gx = Arrays.copyOf(swarm.px[i], PROBLEM_DIMENSION);
                    swarm.gf = swarm.pf[i];
                }

            }
        }
    }

    public static void printPerformance(double...record){

        double[] saveRecord = new double[INITIAL_RECORD.length + record.length];

        System.arraycopy(INITIAL_RECORD, 0, saveRecord, 0, INITIAL_RECORD.length);
        System.arraycopy(record, 0, saveRecord, INITIAL_RECORD.length, record.length);

        System.out.print(String.format(Locale.US, "%.3e", saveRecord[0]));
        for (int i = 1; i < saveRecord.length; i++) {
            System.out.print(String.format(Locale.US, "\t%.3e", saveRecord[i]));
        }
        System.out.println();
    }

    static void printHeadOfOutputFile(){

        try {
            System.setOut(new PrintStream(new File(OUTPUT_FILE)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        INITIAL_RECORD = new double[]{CHANGE_TYPE, FUTURE_HORIZON, MAX_CHANGES, CHANGE_FREQUENCY, ALGORITHM_ID, POPULATION_SIZE, SEED};

        String[] colNames = {"CHANGE_TYPE", "FUTURE_HORIZON", "MAX_CHANGES", "CHANGES_FREQUENCY", "ALGORITHM_ID", "POPULATION_SIZE", "SEED", "RUN_ID", "CHANGE", "ITERATION", "BEST_FITNESS"};

        System.out.print(colNames[0]);
        for (int i = 1; i < colNames.length; i++) {
            System.out.print("\t" + colNames[i]);
        }
        System.out.println();
    }


}
