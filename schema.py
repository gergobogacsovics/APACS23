{
    'base': {
        'required': True,
        'type': 'dict',
        'schema': {
            'model': {
                'required': True,
                'type': 'string'
            },
            'gpu_id': {
                'required': True,
                'type': 'integer'
            },
            'mode': {
                'required': True,
                'type': 'string'
            },
            'hyper_params': {
                'required': True,
                'type': 'dict',
                'schema': {
                    'classes': {
                        'required': True,
                        'type': 'integer'
                    },
                    'batch_size': {
                        'required': True,
                        'type': 'integer'
                    },
                    'pixels_cut': {
                        'required': True,
                        'type': 'integer'
                    }
                }
            }
        }
    },
    'datasets': {
                'required': True,
                'type': 'dict',
                'schema': {
                    'training': {
                        'required': False,
                        'type': 'dict',
                        'schema': {
                            'dir_inputs': {
                                'required': True,
                                'type': 'string'
                            },
                            'dir_masks': {
                                'required': True,
                                'type': 'string'
                            }
                        }
                    },
                    'test': {
                        'required': False,
                        'type': 'dict',
                        'schema': {
                            'dir_inputs': {
                                'required': True,
                                'type': 'string'
                            },
                            'dir_masks': {
                                'required': True,
                                'type': 'string'
                            }
                        }
                    },
                    'validation': {
                        'required': False,
                        'type': 'dict',
                        'schema': {
                            'dir_inputs': {
                                'required': True,
                                'type': 'string'
                            },
                            'dir_masks': {
                                'required': True,
                                'type': 'string'
                            }
                        }
                    }
                }
            },
    'modes': {
        'required': True,
        'type': 'dict',
        'schema': {
            'training': {
                'required': True,
                'type': 'dict',
                'schema': {
                    'hyper_params': {
                        'required': True,
                        'type': 'dict',
                        'schema': {
                            'epochs': {
                                'required': True,
                                'type': 'integer'
                            },
                            'lr': {
                                'required': True,
                                'type': 'float'
                            }
                        }
                    },
                    'checkpoints': {
                        'required': True,
                        'type': 'dict',
                        'schema': {
                            'saving_frequency': {
                                'required': True,
                                'type': 'integer'
                            },
                            'saving_directory': {
                                'required': True,
                                'type': 'string'
                            }
                        }
                    }
                }
            },
            'test': {
                'required': True,
                'type': 'dict',
                'schema': {
                    'checkpoint': {
                        'required': True,
                        'type': 'string'
                    },
                    'saving_directory': {
                        'required': True,
                        'type': 'string'
                    }
                }
            }
        }
    }
}
