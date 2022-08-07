import pytest
import torch
from torchvision.models import resnet18

from easyfsl.methods import AbstractMetaLearner


# pylint: disable=not-callable
class TestAMLEvaluateOnOneTask:
    @staticmethod
    @pytest.mark.parametrize(
        "support_images,support_labels,query_images,query_labels,expected_correct,expected_total",
        [
            (
                torch.ones((5, 3, 28, 28)),
                torch.tensor([0, 0, 0, 0, 1]),
                torch.ones((5, 3, 28, 28)),
                torch.tensor([0, 0, 0, 0, 1]),
                1,
                5,
            ),
        ],
    )
    def test_evaluate_on_one_task_gives_correct_output(
        support_images,
        support_labels,
        query_images,
        query_labels,
        expected_correct,
        expected_total,
        mocker,
    ):
        mocker.patch("torch.Tensor.cuda", new=torch.Tensor.cpu)
        mocker.patch(
            "easyfsl.methods.AbstractMetaLearner.forward",
            return_value=torch.tensor(5 * [[0.25, 0.75]]).cuda(),
        )
        mocker.patch("easyfsl.methods.AbstractMetaLearner.process_support_set")
        model = AbstractMetaLearner(resnet18())
        assert (
            model.evaluate_on_one_task(
                support_images,
                support_labels,
                query_images,
                query_labels,
            )
            == (expected_correct, expected_total)
        )


# pylint: enable=not-callable


class TestAMLAbstractMethods:
    @staticmethod
    def test_forward_raises_error_when_not_implemented():
        with pytest.raises(NotImplementedError):
            model = AbstractMetaLearner(resnet18())
            model(None)

    @staticmethod
    def test_process_support_set_raises_error_when_not_implemented():
        with pytest.raises(NotImplementedError):
            model = AbstractMetaLearner(resnet18())
            model.process_support_set(None, None)


class TestAMLValidate:
    @staticmethod
    def test_validate_returns_accuracy(mocker):
        mocker.patch("easyfsl.methods.AbstractMetaLearner.evaluate", return_value=0.0)
        meta_learner = AbstractMetaLearner(resnet18())
        assert meta_learner.validate(None) == 0.0

    @staticmethod
    def test_validate_updates_best_model_state_if_it_has_best_validation_accuracy(
        mocker,
    ):
        mocker.patch("easyfsl.methods.AbstractMetaLearner.evaluate", return_value=0.5)
        meta_learner = AbstractMetaLearner(resnet18())
        meta_learner.best_validation_accuracy = 0.1
        meta_learner.validate(None)
        assert meta_learner.best_model_state is not None

    @staticmethod
    def test_validate_updates_best_accuracy_if_it_has_best_validation_accuracy(
        mocker,
    ):
        mocker.patch("easyfsl.methods.AbstractMetaLearner.evaluate", return_value=0.5)
        meta_learner = AbstractMetaLearner(resnet18())
        meta_learner.best_validation_accuracy = 0.1
        meta_learner.validate(None)
        assert meta_learner.best_validation_accuracy == 0.5

    @staticmethod
    @pytest.mark.parametrize(
        "accuracy",
        [
            0.05,
            0.1,
        ],
    )
    def test_validate_leaves_best_model_state_if_it_has_worse_validation_accuracy(
        accuracy,
        mocker,
    ):
        mocker.patch(
            "easyfsl.methods.AbstractMetaLearner.evaluate", return_value=accuracy
        )
        meta_learner = AbstractMetaLearner(resnet18())
        meta_learner.best_validation_accuracy = 0.1
        meta_learner.validate(None)
        assert meta_learner.best_model_state is None

    @staticmethod
    @pytest.mark.parametrize(
        "n_train_episodes,validation_frequency,expected_number_of_validations",
        [
            (5, 1, 5),
            (5, 5, 1),
            (5, 6, 0),
            (5, 3, 1),
            (6, 3, 2),
        ],
    )
    def test_validation_occurs_when_expected(
        n_train_episodes, validation_frequency, expected_number_of_validations, mocker
    ):
        mocker.patch(
            "easyfsl.methods.AbstractMetaLearner.fit_on_task", return_value=0.0
        )
        mocker.patch("easyfsl.methods.AbstractMetaLearner.validate")
        spy_validate = mocker.spy(AbstractMetaLearner, "validate")

        meta_learner = AbstractMetaLearner(resnet18())
        train_loader = n_train_episodes * [(None, None, None, None, None)]

        meta_learner.fit(
            train_loader=train_loader,
            optimizer=None,
            val_loader=True,
            validation_frequency=validation_frequency,
        )

        assert spy_validate.call_count == expected_number_of_validations
