// --------------------------------------------------------------------------------------------------------------------
// <summary>
//   The program.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace ProbabilisticMultipleKernelLearning
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;

    using Microsoft.VisualBasic.FileIO;

    using MicrosoftResearch.Infer.Collections;
    using MicrosoftResearch.Infer.Maths;

    /// <summary>
    /// The program.
    /// </summary>
    public class Program
    {
        /// <summary>
        /// The kernel names.
        /// </summary>
        private static readonly string[] KernelNames =
            {
                "Composition", "Secondary", "Hydrophobicity", "Volume", "Polarity", "Polarizability", "L1", "L4",
                "L14", "L30", "SWblosum62", "SWpam50"
            };

        /// <summary>
        /// Defines the entry point of the application.
        /// </summary>
        public static void Main()
        {
            var trainKernels = new Dictionary<string, PositiveDefiniteMatrix>();
            var testKernels = new Dictionary<string, PositiveDefiniteMatrix>();
            var trainLabels = LoadFromCsv("Data\\t_Train.csv").Select(row => int.Parse(row[0])).ToArray();
            var testLabels = LoadFromCsv("Data\\t_Test.csv").Select(row => int.Parse(row[0])).ToArray();

            // Load data
            foreach (string s in KernelNames)
            {
                var train = LoadFromCsv("Data\\" + s + "_Train.csv");
                var test = LoadFromCsv("Data\\" + s + "_Test.csv");
                var data = train.Select(row => row.Select(double.Parse).ToArray()).ToArray();
                trainKernels[s] = new PositiveDefiniteMatrix(data.Length, data[0].Length, data.SelectMany(ia => ia).ToArray());
                data = test.Select(row => row.Select(double.Parse).ToArray()).ToArray();
                testKernels[s] = new PositiveDefiniteMatrix(data.Length, data[0].Length, data.SelectMany(ia => ia).ToArray());
            }

            var trainModel = new MulticlassPmkl();
            var testModel = new MulticlassPmkl();

            var posteriors = trainModel.Train(trainKernels.Values.ToArray(), trainLabels);
            var predictions = testModel.Predict(testKernels.Values.ToArray(), trainLabels.Distinct().ToArray(), posteriors);

            var trainError = predictions.Zip(trainLabels, (p, t) => p == t ? 1.0 : 0.0).Average();
            Console.WriteLine("Train error: " + trainError);
            var testError = predictions.Zip(testLabels, (p, t) => p == t ? 1.0 : 0.0).Average();
            Console.WriteLine("Test error: " + testError);
        }

        /// <summary>
        /// Loads from CSV.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="hasFieldsEnclosedInQuotes">if set to <c>true</c> [has fields enclosed in quotes].</param>
        /// <param name="delimiter">The delimiter.</param>
        /// <param name="verbose">if set to <c>true</c> [verbose].</param>
        /// <param name="encoding">The encoding.</param>
        /// <returns>
        /// The <see cref="IEnumerable{T}" /> of rows.
        /// </returns>
        public static IEnumerable<string[]> LoadFromCsv(
            string filename,
            bool hasFieldsEnclosedInQuotes = false,
            string delimiter = ",",
            bool verbose = false,
            Encoding encoding = null)
        {
            using (FileStream fileStream = new FileStream(filename, FileMode.Open))
            {
                var parser = new TextFieldParser(fileStream, encoding ?? Encoding.Default)
                {
                    TextFieldType = FieldType.Delimited,
                    TrimWhiteSpace = true,
                    HasFieldsEnclosedInQuotes = hasFieldsEnclosedInQuotes,
                };

                parser.SetDelimiters(new[] { delimiter });

                while (!parser.EndOfData)
                {
                    string[] row = null;
                    try
                    {
                        row = parser.ReadFields();
                    }
                    catch (MalformedLineException exception)
                    {
                        if (verbose)
                        {
                            Console.WriteLine(exception.Message);
                        }
                    }

                    if (row == null)
                    {
                        continue;
                    }

                    yield return row;
                }
            }
        }
    }
}
