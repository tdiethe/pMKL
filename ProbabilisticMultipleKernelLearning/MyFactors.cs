// --------------------------------------------------------------------------------------------------------------------
// <copyright file="MyFactors.cs" company="">
//   
// </copyright>
// <summary>
//   Defines the MyFactors type.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace ProbabilisticMultipleKernelLearning
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    using MicrosoftResearch.Infer.Factors;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// The matrix sum.
    /// </summary>
    public static class MyFactors
    {
        /// <summary>
        /// Sums the specified array.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns>
        /// The <see cref="Matrix" />.
        /// </returns>
        /// <exception cref="System.ArgumentNullException">array
        /// or
        /// array;The array must not contain null matrices.</exception>
        /// <exception cref="System.ArgumentException">The array must contain at least one matrix.;array
        /// or
        /// All matrices in the array must have the same number of rows.;array
        /// or
        /// All matrices in the array must have the same number of columns.;array</exception>
        [ParameterNames("MatrixSum", "array")]
        public static Matrix MatrixSum(IList<PositiveDefiniteMatrix> array)
        {
            if (array == null)
            {
                throw new ArgumentNullException("array");
            }

            if (array.Count < 1)
            {
                throw new ArgumentException("The array must contain at least one matrix.", "array");
            }

            if (array.Any(element => element == null))
            {
                throw new ArgumentNullException("array", "The array must not contain null matrices.");
            }

            int rows = array[0].Rows;
            if (array.Any(element => element.Rows != rows))
            {
                throw new ArgumentException("All matrices in the array must have the same number of rows.", "array");
            }

            int cols = array[0].Cols;
            if (array.Any(element => element.Cols != cols))
            {
                throw new ArgumentException("All matrices in the array must have the same number of columns.", "array");
            }

            var sum = Copy(array[0]);
            for (int i = 1; i < array.Count; i++)
            {
                sum.SetToSum(sum, array[i]);
            }

            return sum;
        }

        /// <summary>
        /// Copies the specified matrix.
        /// </summary>
        /// <param name="that">The matrix to copy.</param>
        /// <returns>
        /// The <see cref="Matrix" />.
        /// </returns>
        public static PositiveDefiniteMatrix Copy(PositiveDefiniteMatrix that)
        {
            var m = new PositiveDefiniteMatrix(that.Rows, that.Cols);
            m.SetTo(that);
            return m;
        }
    }
}
