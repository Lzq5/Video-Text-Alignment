class Registry:

    mapping = {
        "dataset_name_mapping": {},
        "model_name_mapping": {},
        "criterion_name_mapping": {},
    }

    @classmethod
    def register_dataset(cls, name):

        def wrap(dataset_cls):
            if name in cls.mapping["dataset_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["dataset_name_mapping"][name]
                    )
                )
            cls.mapping["dataset_name_mapping"][name] = dataset_cls
            return dataset_cls

        return wrap

    @classmethod
    def get_dataset_class(cls, name):
        return cls.mapping["dataset_name_mapping"].get(name, None)


    @classmethod
    def register_model(cls, name):

        def wrap(model_cls):
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def register_criterion(cls, name):

        def wrap(criterion_cls):
            if name in cls.mapping["criterion_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["criterion_name_mapping"][name]
                    )
                )
            cls.mapping["criterion_name_mapping"][name] = criterion_cls
            return criterion_cls

        return wrap

    @classmethod
    def get_criterion_class(cls, name):
        return cls.mapping["criterion_name_mapping"].get(name, None)
    
registry = Registry()
