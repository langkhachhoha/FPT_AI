# Machine Learning in predictioning the atomization energies
In the field of chemistry, understanding molecular stability is very vital. Accurate calculation of electronic properties is essential for the intelligent and rational design of compounds and materials. To achieve this, recently, machine learning is used to model atomization energies of a diverse set of organic molecules, based on nuclear charges and atomic positions only.

In this report, I employ some machine learning algorithms to address the problem: preditioning the atomization energies. The atomization energy E quantifies the potential energy stored in all chemical bonds. As such, it was defined as the difference between the potential energy of a molecule and the sum of potential energies of its composing isolated atoms. 

The experiments were conducted on the QM7 dataset. This is a subset of the GDB-13 database. GDB-13 contains nearly 1 billion stable and synthetically accessible organic molecules. In the QM7 subset, only molecules with up to 23 atoms are included. These atoms consist of carbon (C), nitrogen (N), oxygen (O), and sulfur (S). The total number of molecules in the QM7 dataset is 7165. Each molecule is represented using the Coulomb matrix, which captures the interactions between atoms.

I categorize my algorithms into three approaches:
- Traditional Machine Learning: Linear Regression, Support Vector Regression, AdaBoost, XGBoost.
- Neural Network: Multilayer perceptron (MLP), Convolutional neural network (CNN).
- Graph Neural Network.


