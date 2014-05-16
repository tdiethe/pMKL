// --------------------------------------------------------------------------------------------------------------------
// <summary>
//   The multiclass PMKL.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace ProbabilisticMultipleKernelLearning
{
    using System.Collections.Generic;
    using System.Linq;

    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Models;

    /// <summary>
    /// The multiclass PMKL.
    /// </summary>
    public class MulticlassPmkl
    {
        /// <summary>
        /// The tau.
        /// </summary>
        private readonly Variable<double> tau;

        /// <summary>
        /// The nu.
        /// </summary>
        private readonly Variable<double> nu;

        /// <summary>
        /// The mu.
        /// </summary>
        private readonly Variable<double> mu;

        /// <summary>
        /// The lambda.
        /// </summary>
        private readonly Variable<double> lambda;

        /// <summary>
        /// The zero vector.
        /// </summary>
        private readonly Variable<Vector> zero;

        /// <summary>
        /// The identity matrix.
        /// </summary>
        private readonly Variable<PositiveDefiniteMatrix> identity;

        /// <summary>
        /// The weight precision.
        /// </summary>
        private readonly Variable<PositiveDefiniteMatrix> weightPrecision;

        /// <summary>
        /// The number of instances.
        /// </summary>
        private readonly Variable<int> numberOfInstances;

        /// <summary>
        /// The number of classes.
        /// </summary>
        private readonly Variable<int> numberOfClasses;

        /// <summary>
        /// The number of kernels.
        /// </summary>
        private readonly Variable<int> numberOfKernels;

        /// <summary>
        /// The labels.
        /// </summary>
        private readonly VariableArray<int> labels;

        /// <summary>
        /// The kernels.
        /// </summary>
        private readonly VariableArray<PositiveDefiniteMatrix> kernels;

        /// <summary>
        /// The weights.
        /// </summary>
        private readonly VariableArray<Vector> weights;

        /// <summary>
        /// The beta.
        /// </summary>
        private readonly Variable<Vector> beta;

        /// <summary>
        /// Initializes a new instance of the <see cref="MulticlassPmkl"/> class.
        /// </summary>
        public MulticlassPmkl()
        {
            this.numberOfInstances = Variable.New<int>().Named("numberOfInstances").Attrib(new DoNotInfer());
            Range instance = new Range(this.numberOfInstances).Named("instance");

            this.numberOfClasses = Variable.New<int>().Named("numberOfClasses").Attrib(new DoNotInfer());
            Range cls = new Range(this.numberOfClasses).Named("cls");

            this.numberOfKernels = Variable.New<int>().Named("numberOfKernels").Attrib(new DoNotInfer());
            Range kernel = new Range(this.numberOfKernels).Named("kernel");

            this.zero = Variable.New<Vector>().Named("zero").Attrib(new DoNotInfer());
            this.identity= Variable.New<PositiveDefiniteMatrix>().Named("identity").Attrib(new DoNotInfer());

            // kernels
            this.kernels = Variable.Array<PositiveDefiniteMatrix>(kernel).Named("kernels").Attrib(new DoNotInfer());

            // hyperpriors for alpha
            this.mu = Variable.New<double>().Named("mu").Attrib(new DoNotInfer());
            this.lambda = Variable.New<double>().Named("lambda").Attrib(new DoNotInfer());

            // priors over the betas
            var alpha = Variable.Array<double>(kernel).Named("alpha");
            alpha[kernel] = Variable.GammaFromShapeAndScale(this.mu, this.lambda).ForEach(kernel);

            // betas
            // this.beta = Variable.Dirichlet(kernel, Variable.Vector(alpha)).Named("beta");
            this.beta = Variable.VectorGaussianFromMeanAndPrecision(this.zero, this.identity);

            // hyperpriors for z
            this.tau = Variable.New<double>().Named("tau").Attrib(new DoNotInfer());
            this.nu = Variable.New<double>().Named("nu").Attrib(new DoNotInfer());

            // priors over weight precisions
            //// var z = Variable.Array<double>(cls).Named("z");
            //// z[cls] = Variable.GammaFromShapeAndScale(tau, nu).ForEach(cls);

            // this.weightPrecision = Variable.New<PositiveDefiniteMatrix>().Named("weightPrecision").Attrib(new DoNotInfer());

            // weights
            this.weights = Variable.Array<Vector>(cls).Named("w");
            this.weights[cls] = Variable.VectorGaussianFromMeanAndPrecision(this.zero, this.identity).ForEach(cls);

            // labels
            this.labels = Variable.Array<int>(instance).Named("labels").Attrib(new DoNotInfer());

            // weighted sum kernel
            var scaledKernels = Variable.Array<PositiveDefiniteMatrix>(kernel).Named("scaledKernels");
            using (var k = Variable.ForEach(kernel))
            {
                scaledKernels[kernel] = Variable.MatrixTimesScalar(this.kernels[kernel], Variable.GetItem(this.beta, k.Index));
            }

            var kernelSum = Variable<Matrix>.Factor(MyFactors.MatrixSum, scaledKernels).Named("kernelSum").Attrib(new MarginalPrototype(new Wishart(0)));
            
            var affinity = Variable.Array<Vector>(cls).Named("affinity").Attrib(new MarginalPrototype(new Gaussian()));
            affinity[cls] = Variable.MatrixTimesVector(kernelSum, this.weights[cls]);

            using (var i = Variable.ForEach(instance))
            {
                var aa = Variable.Array<double>(cls).Named("aa");
                aa[cls] = Variable.GetItem(affinity[cls], i.Index);
                var p = Variable.Softmax(aa).Named("p");
                this.labels[instance] = Variable.Discrete(cls, p);
            }
        }

        /// <summary>
        /// Trains the specified train kernels.
        /// </summary>
        /// <param name="trainKernels">The train kernels.</param>
        /// <param name="trainLabels">The train labels.</param>
        /// <returns>
        /// The <see cref="Marginals" />.
        /// </returns>
        public Marginals Train(IList<PositiveDefiniteMatrix> trainKernels, IList<int> trainLabels)
        {
            // Set the hyperpriors
            this.tau.ObservedValue = 1;
            this.nu.ObservedValue = 1;
            this.mu.ObservedValue = 1;
            this.lambda.ObservedValue = 1;

            this.zero.ObservedValue = Vector.Zero(trainLabels.Count);
            this.identity.ObservedValue = PositiveDefiniteMatrix.Identity(trainLabels.Count);
            //// this.weightPrecision.ObservedValue = PositiveDefiniteMatrix.Identity(trainLabels.Count);
            this.numberOfClasses.ObservedValue = trainLabels.Distinct().Count();
            this.numberOfInstances.ObservedValue = trainLabels.Count;
            this.numberOfKernels.ObservedValue = trainKernels.Count;
            this.labels.ObservedValue = trainLabels.ToArray();
            this.kernels.ObservedValue = trainKernels.ToArray();

            var engine = new InferenceEngine { Algorithm = new ExpectationPropagation { DefaultNumberOfIterations = 5 } };
            return new Marginals { Weights = engine.Infer<Gaussian[][]>(this.weights), Beta = engine.Infer<Dirichlet>(this.beta) };
        }

        /// <summary>
        /// Predicts the specified test kernels.
        /// </summary>
        /// <param name="testKernels">The test kernels.</param>
        /// <param name="classes">The classes.</param>
        /// <param name="posteriors">The posteriors.</param>
        /// <returns>
        /// The <see cref="IList{Int32}" />.
        /// </returns>
        public IList<int> Predict(IList<PositiveDefiniteMatrix> testKernels, int[] classes, Marginals posteriors)
        {
            this.numberOfClasses.ObservedValue = classes.Count();
            this.numberOfInstances.ObservedValue = testKernels[0].Rows;
            this.numberOfKernels.ObservedValue = testKernels.Count;
            this.kernels.ObservedValue = testKernels.ToArray();
            this.beta.ObservedValue = posteriors.Beta.GetMean();
            this.weights.ObservedValue = posteriors.Weights.Select(ia => Vector.FromArray(ia.Select(inner => inner.GetMean()).ToArray())).ToArray();

            var engine = new InferenceEngine { Algorithm = new ExpectationPropagation { DefaultNumberOfIterations = 5 } };
            return engine.Infer<IList<int>>(this.labels);    
        }

        /// <summary>
        /// The marginal distributions.
        /// </summary>
        public class Marginals
        {
            /// <summary>
            /// Gets or sets the per-class weights.
            /// </summary>
            public Gaussian[][] Weights { get; set; }

            /// <summary>
            /// Gets or sets the beta.
            /// </summary>
            public Dirichlet Beta { get; set; }
        }
    }
}
