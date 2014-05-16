// --------------------------------------------------------------------------------------------------------------------
// <copyright file="MatrixSumOp.cs" company="">
//   
// </copyright>
// <summary>
//   Defines the MatrixSumOp type.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace ProbabilisticMultipleKernelLearning
{
    using System.Collections.Generic;
    using System.Runtime.InteropServices;

    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Factors;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// The matrix sum op.
    /// </summary>
    [FactorMethod(typeof(MyFactors), "MatrixSum")]
    public static class MatrixSumOp
    {
        /// <summary>
        /// Sums the average conditional.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns>
        /// The <see cref="Wishart" />.
        /// </returns>
        public static Wishart SumAverageConditional([SkipIfUniform] IList<Wishart> array)
        {
            return default(Wishart);
        }

        /// <summary>
        /// Array average conditional.
        /// </summary>
        /// <typeparam name="WishartArray">The type of the array.</typeparam>
        /// <param name="array">The array.</param>
        /// <param name="sum">The sum.</param>
        /// <param name="toSum">To sum.</param>
        /// <param name="result">The result.</param>
        /// <returns>
        /// The <see cref="WishartArray" />.
        /// </returns>
        public static WishartArray ArrayAverageConditional<WishartArray>([SkipIfUniform] WishartArray array, [SkipIfUniform] Wishart sum, [Fresh] Wishart toSum, WishartArray result)
          where WishartArray : IList<Wishart>
        {
            var dimension = toSum.Dimension;

            var mean = new PositiveDefiniteMatrix(dimension, dimension);
            var variance = new PositiveDefiniteMatrix(dimension, dimension);
            var incomingMean = new PositiveDefiniteMatrix(dimension, dimension);
            var incomingVariance = new PositiveDefiniteMatrix(dimension, dimension);

            // get the mean and variance of sum of all the Wisharts
            toSum.GetMeanAndVariance(mean, variance);

            // subtract it off from the mean and variance of incoming Wishart from Sum
            sum.GetMeanAndVariance(incomingMean, incomingVariance);

            // regularizer to ensure Positive definiteness
            var regularizer = PositiveDefiniteMatrix.IdentityScaledBy(dimension, 1e-5);

            mean = (PositiveDefiniteMatrix)(incomingMean - mean + regularizer);
            variance = incomingVariance + variance;

            for (int i = 0; i < array.Count; i++)
            {
                array[i].GetMeanAndVariance(incomingMean, incomingVariance);
                result[i] = new Wishart(dimension);
                result[i].SetMeanAndVariance(mean + incomingMean, (PositiveDefiniteMatrix)(variance - incomingVariance + regularizer));
            }

            return result;
        }

        /// <summary>
        /// Array average conditional.
        /// </summary>
        /// <typeparam name="WishartArray">The type of the array.</typeparam>
        /// <param name="array">The array.</param>
        /// <param name="sum">The sum.</param>
        /// <param name="result">The result.</param>
        /// <returns>
        /// The <see cref="WishartArray" />.
        /// </returns>
        public static WishartArray ArrayAverageConditional<WishartArray>([SkipIfUniform] WishartArray array, PositiveDefiniteMatrix sum, WishartArray result)
            where WishartArray : IList<Wishart>
        {
            var toSum = SumAverageConditional(array);
            return ArrayAverageConditional(array, Wishart.PointMass(sum), toSum, result);
        }

        /// <summary>
        /// Log average for the factor.
        /// </summary>
        /// <param name="sum">The sum.</param>
        /// <param name="array">The array.</param>
        /// <returns>
        /// The <see cref="double" />.
        /// </returns>
        public static double LogAverageFactor(PositiveDefiniteMatrix sum, [SkipIfUniform] IList<Wishart> array)
        {
            var toSum = SumAverageConditional(array);
            return toSum.GetLogProb(sum);
        }

        /// <summary>
        /// Log average for the factor.
        /// </summary>
        /// <param name="sum">The sum.</param>
        /// <param name="toSum">To sum.</param>
        /// <param name="array">The array.</param>
        /// <returns>
        /// The <see cref="double" />.
        /// </returns>
        public static double LogAverageFactor([SkipIfUniform] Wishart sum, [Fresh] Wishart toSum, [SkipIfUniform] IList<Wishart> array)
        {
            return toSum.GetLogAverageOf(sum);
        }

        /// <summary>
        /// The log evidence ratio.
        /// </summary>
        /// <param name="sum">The sum.</param>
        /// <param name="array">The array.</param>
        /// <returns>
        /// The <see cref="double" />.
        /// </returns>
        public static double LogEvidenceRatio(PositiveDefiniteMatrix sum, IList<Wishart> array)
        {
            return LogAverageFactor(sum, array);
        }

        /// <summary>
        /// The log evidence ratio.
        /// </summary>
        /// <param name="sum">The sum.</param>
        /// <returns>
        /// The <see cref="double" />.
        /// </returns>
        [Skip]
        public static double LogEvidenceRatio(Wishart sum)
        {
            return 0.0;
        } 
    }
}
