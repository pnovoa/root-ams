package org.rootams;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RMPBI {

    public class Environment {

        double[][] H = new double[dimension][numPeaks];
        double[][] W = new double[dimension][numPeaks];
        double[][] C = new double[dimension][numPeaks];
        double S = initialAngle;
        int timeStep;
    }

    public interface ChangeType {

        void change(Environment env);

    }

    public final int learningPeriod=20;

    public final int dimension = 2;
    final int numPeaks = 5;

    public final double minCoord = -25.;
    public final double maxCoord = 25.;

    // heights
    final double minH = 30.;
    final double maxH = 70.;
    final double sevH = 5.;
    // width
    final double minW = 1.;
    final double maxW = 13.;
    final double sevW = 0.5;
    // angle
    final double minS = -Math.PI;
    final double maxS = Math.PI;
    final double sevS = 1.0;

    final double initialAngle = 0.;

    // chaotic constant
    final double A = 3.67;
    // gamma
    final double gamma = 0.04;
    // gamma max
    final double gammaMax = 0.1;
    // period
    final int period = 12;
    // noisy severity
    final double noisySev = 0.8;
    // time windows
    public int timeWindows = 2;
    // computational budget (delta e)
    public int computationalBudget = 2500;

    // number of changes
    public int numChanges = 100;
    public ArrayList<Environment> environments;
    public int seed = 22;
    Random rand;

    ChangeType changeFunction;

    public int currEnvironment;

    public int changeType;

    public void init() {


        switch (changeType) {
            case 1:
                changeFunction = new SmallStepChangeType();
                break;
            case 2:
                changeFunction = new LargeStepChangeType();
                break;
            case 3:
                changeFunction = new RandomChangeType();
                break;
            case 4:
                changeFunction = new ChaoticChangeType();
                break;
            case 5:
                changeFunction = new RecurrentChangeType();
                break;
            default:
                changeFunction = new RecurrentWithNoiseChangeType();
                break;
        }


        environments = new ArrayList<>(numChanges + timeWindows);
        rand = new Random(seed);

        currEnvironment = 0;

        // creating the environments
        Environment env0 = new Environment();
        env0.timeStep = 0;
        environments.add(env0);

        for (int d = 0; d < dimension; d++) {

            for (int i = 0; i < numPeaks; i++) {
                env0.C[d][i] = minCoord + (maxCoord - minCoord) * rand.nextDouble();
                env0.H[d][i] = minH + (maxH - minH) * rand.nextDouble();
                env0.W[d][i] = minW + (maxW - minW) * rand.nextDouble();
            }
        }

        for (int i = 1; i < numChanges + timeWindows; i++) {

            Environment envi = new Environment();
            envi.timeStep = i;

            changeFunction.change(envi);

            environments.add(envi);
        }
    }

    public void change() {

        currEnvironment++;
    }

    /**
     * @param x the solution vector to evaluate
     * @return the function evaluation for x at the current environment
     */
    public double eval(double[] x) {

        return evalEnv(environments.get(currEnvironment), x);
    }

    public double evalEnv(Environment env, double[] x) {

        double result = 0.;

        for (int d = 0; d < dimension; d++) {

            double maxPeak = Double.NEGATIVE_INFINITY;

            for (int i = 0; i < numPeaks; i++) {

                double peak = env.H[d][i] - env.W[d][i] * Math.abs(env.C[d][i] - x[d]);

                if (peak > maxPeak)
                    maxPeak = peak;
            }



            result += maxPeak;
        }

        return result / (double) dimension;

    }

    public double trueEval(double[] x) {

        double result = 0.;

        for(int i=0; i<timeWindows; i++) {

            result += evalEnv(environments.get(currEnvironment+i),x);
        }

        return result/(double)timeWindows;
    }

    public double assistedEval(double present, double future[]){

        double result = present;

        for(int i=0; i<timeWindows-1; i++) {

            result += future[i];
        }

        return result/(double)timeWindows;
    }

    public abstract class AbstractRotationalChangeType implements ChangeType {

        @Override
        public void change(Environment env) {

            specificChange(env);
            rotateCoordinates(env);
        }

        private void rotateCoordinates(Environment env) {

            RealMatrix rotMatrix = MatrixUtils.createRealMatrix(
                    new double[][] { { Math.cos(env.S), -Math.sin(env.S) }, { Math.sin(env.S), Math.cos(env.S) } });

            Environment prevEnv = environments.get(env.timeStep - 1);

            for (int i = 0; i < numPeaks; i++) {
                double[] centersi = { prevEnv.C[0][i], prevEnv.C[1][i] };
                RealMatrix centerV = MatrixUtils.createRowRealMatrix(centersi);
                RealMatrix rotv = centerV.multiply(rotMatrix);
                double x = Utils.clampValue(rotv.getEntry(0, 0), minCoord, maxCoord);
                env.C[0][i] = x;
                x = Utils.clampValue(rotv.getEntry(0, 1), minCoord, maxCoord);
                env.C[1][i] = x;
            }
        }

        public void specificChange(Environment env) {
            Environment prevEnv = environments.get(env.timeStep - 1);

            for (int d = 0; d < dimension; d++) {

                for (int i = 0; i < numPeaks; i++) {
                    env.H[d][i] = changeSingleValue(prevEnv.H[d][i], minH, maxH, sevH);
                    env.W[d][i] = changeSingleValue(prevEnv.W[d][i], minW, maxW, sevW);
                }
            }

            env.S = changeSingleValue(prevEnv.S, minS, maxS, sevS);
        }

        protected abstract double changeSingleValue(double previousV, double minV, double maxV, double sevV);
    }

    public class SmallStepChangeType extends AbstractRotationalChangeType {

        @Override
        protected double changeSingleValue(double previousV, double minV, double maxV, double sevV) {

            double result = previousV + gamma * (maxV - minV) * sevV * (2 * rand.nextDouble() - 1);

            result = Utils.clampValue(result, minV, maxV);

            return result;
        }
    }


    public class LargeStepChangeType extends AbstractRotationalChangeType {

        @Override
        protected double changeSingleValue(double previousV, double minV, double maxV, double sevV) {

            double result = 2*rand.nextDouble()-1;

            result = previousV + (maxV-minV)*(gamma*Math.signum(result) + (gammaMax - gamma)*result)*sevV;

            result = Utils.clampValue(result, minV, maxV);

            return result;
        }


    }

    public class RandomChangeType extends AbstractRotationalChangeType {

        @Override
        protected double changeSingleValue(double previousV, double minV, double maxV, double sevV) {

            double result = previousV + rand.nextGaussian()*sevV;

            result = Utils.clampValue(result, minV, maxV);

            return result;

        }

    }

    public class ChaoticChangeType implements ChangeType{


        public void change(Environment env) {
            Environment prevEnv = environments.get(env.timeStep - 1);

            for (int d = 0; d < dimension; d++) {

                for (int i = 0; i < numPeaks; i++) {
                    env.H[d][i] = changeSingleValue(prevEnv.H[d][i], minH, maxH);
                    env.W[d][i] = changeSingleValue(prevEnv.W[d][i], minW, maxW);
                    env.C[d][i] = changeSingleValue(prevEnv.C[d][i], minCoord, maxCoord);
                }
            }
        }


        protected double changeSingleValue(double previousV, double minV, double maxV) {

            double result = minV + A*(previousV-minV)*(1-(previousV-minV)/(maxV-minV)) ;

            result = Utils.clampValue(result, minV, maxV);

            return result;
        }

    }


    public class RecurrentChangeType extends AbstractRotationalChangeType {

        @Override
        protected double changeSingleValue(double previousV, double minV, double maxV, double angle) {

            double result =  minV + (maxV-minV) *(Math.sin(2*Math.PI*currEnvironment/(double)period + angle) + 1)/2.;

            result = Utils.clampValue(result, minV, maxV);

            return result;
        }

        @Override
        public void specificChange(Environment env) {
            Environment prevEnv = environments.get(env.timeStep - 1);

            for (int d = 0; d < dimension; d++) {

                for (int i = 0; i < numPeaks; i++) {
                    double angle =(double)period*(i+d)/(dimension+numPeaks);
                    env.H[d][i] = changeSingleValue(prevEnv.H[d][i], minH, maxH, angle);
                    env.W[d][i] = changeSingleValue(prevEnv.W[d][i], minW, maxW, angle);
                }
            }

            env.S = 2*Math.PI/period;
        }

    }

    public class RecurrentWithNoiseChangeType extends RecurrentChangeType {

        @Override
        protected double changeSingleValue(double previousV, double minV, double maxV, double angle) {

            double result  = super.changeSingleValue(previousV, minV, maxV, angle)+ noisySev*rand.nextGaussian();

            result = Utils.clampValue(result, minV, maxV);

            return result;
        }

    }


    public double[] findBestSolution(int timeStep) {

        double[] result = new double[dimension];

        List<Environment> envs = environments.subList(timeStep, timeStep+timeWindows);

        //double MAFFitness = 0.;

        for(int d=0; d<dimension; d++) {

            double MAFdim = Double.NEGATIVE_INFINITY;
            double centMAF = 0;

            for(int i=0; i<numPeaks; i++) {

                double[] MAFPeak = findBestCenterMAFpeak(envs, i, d);

                if(MAFPeak[1]>MAFdim) {

                    MAFdim = MAFPeak[1];
                    centMAF = MAFPeak[0];
                }

            }

            //MAFFitness += MAFdim;
            result[d] = centMAF;
        }

        //MAFFitness = MAFFitness/(double)dimension;
        //result.setFitness(MAFFitness);

        return result;
    }

    protected double[] findBestCenterMAFpeak(List<Environment> arrEnv, int peak, int dim) {

        double bestMAF=Double.NEGATIVE_INFINITY;
        double bestC = 0;

        for(Environment env : arrEnv) {

            double MAFpeak = env.H[dim][peak];

            for(Environment env2 : arrEnv) {

                if(!env2.equals(env)) {

                    MAFpeak += peakEval(env2, env.C[dim][peak], peak, dim);
                }
            }

            MAFpeak = MAFpeak/(double)arrEnv.size();

            if(MAFpeak>bestMAF) {
                bestMAF = MAFpeak;
                bestC = env.C[dim][peak];
            }
        }

        double[] realSol = new double[2];
        realSol[0] = bestC;
        realSol[1] = bestMAF;
        return realSol;
    }

    protected double peakEval(Environment env, double x, int peak, int dim) {

        return env.H[dim][peak] - env.W[dim][peak]*Math.abs(env.C[dim][peak] - x);

    }

}
