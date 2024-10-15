class BatchEvaluator:
    def __init__(self):
        self.all_true_comp_tuples = []
        self.all_pred_comp_tuples = []
        self.all_true_rel_tuples = []
        self.all_pred_rel_tuples = []
        self.macro_f1_components = []
        self.macro_f1_relations = []

    def add_batch(self, true_comp, true_rel, pred_comp, pred_rel):
        for i in range(len(true_comp)):
            self.add_single(true_comp[i], true_rel[i], pred_comp[i], pred_rel[i])

    def add_single(self, true_comp_tuples, true_rel_tuples, pred_comp_tuples, pred_rel_tuples):
        # Ensure the tuples are hashable by converting sets/lists to tuples if necessary
        true_comp_tuples = [tuple(comp) for comp in true_comp_tuples]
        pred_comp_tuples = [tuple(comp) for comp in pred_comp_tuples]
        true_rel_tuples = [tuple(rel) for rel in true_rel_tuples]
        pred_rel_tuples = [tuple(rel) for rel in pred_rel_tuples]

        self.all_true_comp_tuples.extend(true_comp_tuples)
        self.all_pred_comp_tuples.extend(pred_comp_tuples)
        self.all_true_rel_tuples.extend(true_rel_tuples)
        self.all_pred_rel_tuples.extend(pred_rel_tuples)

        # Calculate F1 for components for this instance and store it
        comp_correct = set(true_comp_tuples) & set(pred_comp_tuples)
        comp_precision = len(comp_correct) / len(pred_comp_tuples) if pred_comp_tuples else 0
        comp_recall = len(comp_correct) / len(true_comp_tuples) if true_comp_tuples else 0
        comp_f1 = (2 * comp_precision * comp_recall / (comp_precision + comp_recall)) if (comp_precision + comp_recall) else 0
        self.macro_f1_components.append(comp_f1)

        # Calculate F1 for relations for this instance and store it
        rel_correct = set(true_rel_tuples) & set(pred_rel_tuples)
        rel_precision = len(rel_correct) / len(pred_rel_tuples) if pred_rel_tuples else 0
        rel_recall = len(rel_correct) / len(true_rel_tuples) if true_rel_tuples else 0
        rel_f1 = (2 * rel_precision * rel_recall / (rel_precision + rel_recall)) if (rel_precision + rel_recall) else 0
        self.macro_f1_relations.append(rel_f1)

        return {
            'component_precision': comp_precision,
            'component_recall': comp_recall,
            'component_f1': comp_f1,
            'relation_precision': rel_precision,
            'relation_recall': rel_recall,
            'relation_f1': rel_f1
        }

    def evaluate(self):
        # Convert to sets to remove duplicates and enable direct comparison
        all_true_comp_tuples = list(set(self.all_true_comp_tuples))
        all_pred_comp_tuples = list(set(self.all_pred_comp_tuples))
        all_true_rel_tuples = list(set(self.all_true_rel_tuples))
        all_pred_rel_tuples = list(set(self.all_pred_rel_tuples))

        # Calculate precision, recall, and F1 for components (micro)
        correct_comp_tuples = set(all_true_comp_tuples) & set(all_pred_comp_tuples)
        comp_precision = len(correct_comp_tuples) / len(all_pred_comp_tuples) if all_pred_comp_tuples else 0
        comp_recall = len(correct_comp_tuples) / len(all_true_comp_tuples) if all_true_comp_tuples else 0
        comp_f1 = (2 * comp_precision * comp_recall / (comp_precision + comp_recall)) if (comp_precision + comp_recall) else 0

        # Calculate precision, recall, and F1 for relations (micro)
        correct_rel_tuples = set(all_true_rel_tuples) & set(all_pred_rel_tuples)
        rel_precision = len(correct_rel_tuples) / len(all_pred_rel_tuples) if all_pred_rel_tuples else 0
        rel_recall = len(correct_rel_tuples) / len(all_true_rel_tuples) if all_true_rel_tuples else 0
        rel_f1 = (2 * rel_precision * rel_recall / (rel_precision + rel_recall)) if (rel_precision + rel_recall) else 0

        # Calculate macro F1 by averaging individual F1 scores
        macro_f1_components = sum(self.macro_f1_components) / len(self.macro_f1_components) if self.macro_f1_components else 0

        return {
            'component_precision': comp_precision,
            'component_recall': comp_recall,
            'component_f1': comp_f1,
            'relation_precision': rel_precision,
            'relation_recall': rel_recall,
            'relation_f1': rel_f1,
            'component_macro_f1': macro_f1_components
        }
