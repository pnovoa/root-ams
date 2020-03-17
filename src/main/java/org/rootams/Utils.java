package org.rootams;

public class Utils {

    public static double clampValue(double val, double minV, double maxV) {

        double result = Math.max(val, minV);
        result = Math.min(result, maxV);

        return result;
    }

    public static double[] appendValueToVector(double value, double[] vector){

        double[] result = new double[vector.length+1];

        System.arraycopy(vector, 0, result, 0, vector.length);

        result[vector.length] = value;

        return result;
    }

    public static double[] appendVectorToVector(double[] source, double[] destiny){

        double[] result = new double[source.length + destiny.length];

        System.arraycopy(destiny, 0, result, 0, destiny.length);

        System.arraycopy(source, 0, result, destiny.length, source.length);

        return result;
    }


    public static double[][] appendMatrixToMatrix(double[][] source, double[][] destiny){

        int dimension = source[0].length;
        double[][] result = new double[source.length + destiny.length][dimension];

        for (int i = 0; i < destiny.length; i++) {
            System.arraycopy(destiny[i], 0, result[i], 0, dimension);
        }
        for (int i = 0; i < source.length; i++) {
            System.arraycopy(source[i], 0, result[i+destiny.length], 0, dimension);
        }

        return result;
    }


}