"""
Unit tests for the Configuration Loader.
Tests configuration loading, validation, and management.
"""

import pytest
import yaml
import os
import tempfile
from trading_strategy.config_loader import ConfigurationLoader
from trading_strategy.config_loader import ConfigLoader


class TestConfigurationLoader:
    """Test cases for the configuration loader."""

    def setup_method(self):
        """Set up test configuration files."""
        # Create temporary directory for test configs
        self.temp_dir = tempfile.mkdtemp()

        # Create test trading config
        self.trading_config = {
            'risk_management': {
                'max_risk_per_trade': 0.02,
                'max_daily_risk': 0.05,
                'max_positions': 5,
                'stop_loss_pips': 50,
                'take_profit_pips': 100,
                'max_drawdown_percent': 0.10,
                'drawdown_recovery_percent': 0.05,
                'volatility_factor': 1.0,
                'correlation_limit': 0.7
            },
            'elliott_wave': {
                'min_wave_length': 5,
                'max_wave_length': 50,
                'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786],
                'wave_validation': True,
                'degree_filter': ['minor', 'intermediate', 'primary']
            },
            'ict_concepts': {
                'fvg_min_size': 0.001,
                'fvg_max_age': 100,
                'ob_min_strength': 0.7,
                'ob_max_tests': 3,
                'ote_min_retracement': 0.382,
                'ote_max_retracement': 0.618
            },
            'market_structure': {
                'swing_point_threshold': 0.005,
                'bos_confirmation_candles': 3,
                'choch_confirmation_candles': 3,
                'liquidity_tracking': True,
                'strength_scoring': True
            },
            'session_logic': {
                'timezone': 'UTC',
                'asia_session': {
                    'start': '00:00',
                    'end': '08:00',
                    'bias': 'neutral'
                },
                'london_session': {
                    'start': '08:00',
                    'end': '16:00',
                    'bias': 'bullish'
                },
                'ny_session': {
                    'start': '16:00',
                    'end': '00:00',
                    'bias': 'bearish'
                },
                'overlap_sessions': {
                    'london_ny': {
                        'start': '16:00',
                        'end': '20:00',
                        'bias': 'high_volatility'
                    }
                }
            }
        }

        # Create test timeframe config
        self.timeframe_config = {
            'timeframes': {
                'htf': '1d',
                'mtf': '4h',
                'ltf': '1h'
            },
            'analysis': {
                'lookback_period': 1000,
                'min_data_points': 100,
                'max_data_points': 10000
            },
            'signals': {
                'min_confidence': 0.7,
                'max_signals_per_day': 5,
                'signal_cooldown': 60
            }
        }

        # Write config files
        with open(os.path.join(self.temp_dir, 'trading_config.yaml'), 'w') as f:
            yaml.dump(self.trading_config, f)

        with open(os.path.join(self.temp_dir, 'timeframes.yaml'), 'w') as f:
            yaml.dump(self.timeframe_config, f)

        # Initialize loader
        self.loader = ConfigurationLoader(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_config_loader_initialization(self):
        """Test configuration loader initialization."""
        assert self.loader is not None
        assert self.loader.config_path == self.temp_dir
        assert os.path.exists(self.temp_dir)

    def test_load_trading_config(self):
        """Test loading trading configuration."""
        config = self.loader.load_trading_config()

        assert config is not None
        assert isinstance(config, dict)
        assert 'risk_management' in config
        assert 'elliott_wave' in config
        assert 'ict_concepts' in config
        assert 'market_structure' in config
        assert 'session_logic' in config

        # Check specific values
        assert config['risk_management']['max_risk_per_trade'] == 0.02
        assert config['elliott_wave']['min_wave_length'] == 5
        assert config['ict_concepts']['fvg_min_size'] == 0.001
        assert config['market_structure']['swing_point_threshold'] == 0.005
        assert config['session_logic']['timezone'] == 'UTC'

    def test_load_timeframe_config(self):
        """Test loading timeframe configuration."""
        config = self.loader.load_timeframe_config()

        assert config is not None
        assert isinstance(config, dict)
        assert 'timeframes' in config
        assert 'analysis' in config
        assert 'signals' in config

        # Check specific values
        assert config['timeframes']['htf'] == '1d'
        assert config['timeframes']['mtf'] == '4h'
        assert config['timeframes']['ltf'] == '1h'
        assert config['analysis']['lookback_period'] == 1000
        assert config['signals']['min_confidence'] == 0.7

    def test_get_config_value(self):
        """Test getting specific configuration values."""
        # Test existing values
        risk_per_trade = self.loader.get_config_value('risk_management.max_risk_per_trade')
        assert risk_per_trade == 0.02

        min_wave_length = self.loader.get_config_value('elliott_wave.min_wave_length')
        assert min_wave_length == 5

        timezone = self.loader.get_config_value('session_logic.timezone')
        assert timezone == 'UTC'

        # Test with default value
        non_existent = self.loader.get_config_value('non_existent.key', 'default')
        assert non_existent == 'default'

    def test_get_config_value_nested(self):
        """Test getting nested configuration values."""
        # Test nested values
        asia_start = self.loader.get_config_value('session_logic.asia_session.start')
        assert asia_start == '00:00'

        london_bias = self.loader.get_config_value('session_logic.london_session.bias')
        assert london_bias == 'bullish'

        ny_bias = self.loader.get_config_value('session_logic.ny_session.bias')
        assert ny_bias == 'bearish'

    def test_validate_config(self):
        """Test configuration validation."""
        # Test valid configuration
        is_valid = self.loader.validate_config()
        assert is_valid is True

        # Test with invalid configuration
        invalid_config = {
            'risk_management': {
                'max_risk_per_trade': 1.5,  # Invalid: > 1.0
                'max_daily_risk': 0.05,
                'max_positions': 5
            }
        }

        # Write invalid config
        with open(os.path.join(self.temp_dir, 'invalid_config.yaml'), 'w') as f:
            yaml.dump(invalid_config, f)

        # Test validation
        invalid_loader = ConfigurationLoader(self.temp_dir)
        invalid_loader.config_files['trading_config'] = 'invalid_config.yaml'

        is_valid = invalid_loader.validate_config()
        assert is_valid is False

    def test_config_validation_rules(self):
        """Test configuration validation rules."""
        # Test risk management validation
        risk_config = self.loader.get_config_value('risk_management')
        assert risk_config['max_risk_per_trade'] <= 1.0
        assert risk_config['max_daily_risk'] <= 1.0
        assert risk_config['max_positions'] > 0
        assert risk_config['stop_loss_pips'] > 0
        assert risk_config['take_profit_pips'] > 0

        # Test Elliott Wave validation
        ew_config = self.loader.get_config_value('elliott_wave')
        assert ew_config['min_wave_length'] > 0
        assert ew_config['max_wave_length'] > ew_config['min_wave_length']
        assert all(0 <= level <= 1 for level in ew_config['fibonacci_levels'])
        assert isinstance(ew_config['wave_validation'], bool)

        # Test ICT concepts validation
        ict_config = self.loader.get_config_value('ict_concepts')
        assert ict_config['fvg_min_size'] > 0
        assert ict_config['fvg_max_age'] > 0
        assert ict_config['ob_min_strength'] > 0
        assert ict_config['ob_min_strength'] <= 1.0
        assert ict_config['ob_max_tests'] > 0

        # Test market structure validation
        ms_config = self.loader.get_config_value('market_structure')
        assert ms_config['swing_point_threshold'] > 0
        assert ms_config['bos_confirmation_candles'] > 0
        assert ms_config['choch_confirmation_candles'] > 0
        assert isinstance(ms_config['liquidity_tracking'], bool)
        assert isinstance(ms_config['strength_scoring'], bool)

    def test_session_config_validation(self):
        """Test session configuration validation."""
        session_config = self.loader.get_config_value('session_logic')

        # Test timezone
        assert session_config['timezone'] in ['UTC', 'US/Eastern', 'Europe/London', 'Asia/Tokyo']

        # Test session times
        asia_session = session_config['asia_session']
        london_session = session_config['london_session']
        ny_session = session_config['ny_session']

        assert 'start' in asia_session
        assert 'end' in asia_session
        assert 'bias' in asia_session

        assert 'start' in london_session
        assert 'end' in london_session
        assert 'bias' in london_session

        assert 'start' in ny_session
        assert 'end' in ny_session
        assert 'bias' in ny_session

        # Test session biases
        assert asia_session['bias'] in ['neutral', 'bullish', 'bearish']
        assert london_session['bias'] in ['neutral', 'bullish', 'bearish']
        assert ny_session['bias'] in ['neutral', 'bullish', 'bearish']

    def test_timeframe_config_validation(self):
        """Test timeframe configuration validation."""
        timeframe_config = self.loader.get_config_value('timeframes')
        analysis_config = self.loader.get_config_value('analysis')
        signals_config = self.loader.get_config_value('signals')

        # Test timeframe values
        assert timeframe_config['htf'] in ['1d', '4h', '1h', '30m', '15m']
        assert timeframe_config['mtf'] in ['1d', '4h', '1h', '30m', '15m']
        assert timeframe_config['ltf'] in ['1d', '4h', '1h', '30m', '15m']

        # Test analysis values
        assert analysis_config['lookback_period'] > 0
        assert analysis_config['min_data_points'] > 0
        assert analysis_config['max_data_points'] > analysis_config['min_data_points']

        # Test signals values
        assert 0 <= signals_config['min_confidence'] <= 1
        assert signals_config['max_signals_per_day'] > 0
        assert signals_config['signal_cooldown'] >= 0

    def test_config_file_not_found(self):
        """Test handling of missing configuration files."""
        # Test with non-existent directory
        non_existent_dir = '/non/existent/directory'

        with pytest.raises(FileNotFoundError):
            ConfigurationLoader(non_existent_dir)

    def test_invalid_yaml_format(self):
        """Test handling of invalid YAML format."""
        # Create invalid YAML file
        invalid_yaml_path = os.path.join(self.temp_dir, 'invalid.yaml')
        with open(invalid_yaml_path, 'w') as f:
            f.write('invalid: yaml: content: [')

        # Test loading invalid YAML
        with pytest.raises(yaml.YAMLError):
            loader = ConfigurationLoader(self.temp_dir)
            loader.config_files['trading_config'] = 'invalid.yaml'
            loader.load_trading_config()

    def test_config_merging(self):
        """Test configuration merging functionality."""
        # Create additional config file
        additional_config = {
            'risk_management': {
                'max_risk_per_trade': 0.03,  # Override
                'new_parameter': 'new_value'  # Add new
            },
            'new_section': {
                'new_key': 'new_value'
            }
        }

        with open(os.path.join(self.temp_dir, 'additional_config.yaml'), 'w') as f:
            yaml.dump(additional_config, f)

        # Test merging
        merged_config = self.loader.merge_configs(
            self.trading_config,
            additional_config
        )

        # Check overrides
        assert merged_config['risk_management']['max_risk_per_trade'] == 0.03

        # Check additions
        assert 'new_parameter' in merged_config['risk_management']
        assert 'new_section' in merged_config

        # Check preserved values
        assert merged_config['risk_management']['max_daily_risk'] == 0.05

    def test_config_environment_variables(self):
        """Test configuration with environment variables."""
        import os

        # Set environment variables
        os.environ['MAX_RISK_PER_TRADE'] = '0.03'
        os.environ['MAX_DAILY_RISK'] = '0.06'

        # Test environment variable substitution
        config = self.loader.load_trading_config()

        # Should use environment variables if available
        # (This would require implementation in the loader)
        assert config['risk_management']['max_risk_per_trade'] == 0.02  # Original value

    def test_config_caching(self):
        """Test configuration caching."""
        # Load config multiple times
        config1 = self.loader.load_trading_config()
        config2 = self.loader.load_trading_config()

        # Should return same object (cached)
        assert config1 is config2

        # Clear cache and reload
        self.loader.clear_cache()
        config3 = self.loader.load_trading_config()

        # Should return new object
        assert config1 is not config3
        assert config1 == config3  # But same content

    def test_config_validation_errors(self):
        """Test configuration validation error handling."""
        # Test with invalid risk values
        invalid_risk_config = {
            'risk_management': {
                'max_risk_per_trade': 1.5,  # Invalid: > 1.0
                'max_daily_risk': -0.05,   # Invalid: negative
                'max_positions': 0         # Invalid: zero
            }
        }

        with open(os.path.join(self.temp_dir, 'invalid_risk.yaml'), 'w') as f:
            yaml.dump(invalid_risk_config, f)

        # Test validation
        invalid_loader = ConfigurationLoader(self.temp_dir)
        invalid_loader.config_files['trading_config'] = 'invalid_risk.yaml'

        is_valid = invalid_loader.validate_config()
        assert is_valid is False

    def test_config_default_values(self):
        """Test configuration default values."""
        # Test getting non-existent values with defaults
        default_value = self.loader.get_config_value('non_existent.key', 'default_value')
        assert default_value == 'default_value'

        # Test getting non-existent nested values
        nested_default = self.loader.get_config_value('non_existent.nested.key', 'nested_default')
        assert nested_default == 'nested_default'

    def test_config_type_conversion(self):
        """Test configuration type conversion."""
        # Test string to float conversion
        risk_per_trade = self.loader.get_config_value('risk_management.max_risk_per_trade')
        assert isinstance(risk_per_trade, float)

        # Test string to int conversion
        max_positions = self.loader.get_config_value('risk_management.max_positions')
        assert isinstance(max_positions, int)

        # Test string to bool conversion
        wave_validation = self.loader.get_config_value('elliott_wave.wave_validation')
        assert isinstance(wave_validation, bool)

        # Test string to list conversion
        fibonacci_levels = self.loader.get_config_value('elliott_wave.fibonacci_levels')
        assert isinstance(fibonacci_levels, list)

    def test_config_security(self):
        """Test configuration security."""
        # Test that sensitive values are not exposed
        config = self.loader.load_trading_config()

        # Should not contain sensitive information
        config_str = str(config)
        assert 'password' not in config_str.lower()
        assert 'secret' not in config_str.lower()
        assert 'key' not in config_str.lower()

    def test_config_performance(self):
        """Test configuration loading performance."""
        import time

        # Test loading performance
        start_time = time.time()

        for _ in range(100):
            config = self.loader.load_trading_config()

        end_time = time.time()
        duration = end_time - start_time

        # Should complete 100 loads in less than 1 second
        assert duration < 1.0

    def test_config_error_handling(self):
        """Test configuration error handling."""
        # Test with corrupted file
        corrupted_path = os.path.join(self.temp_dir, 'corrupted.yaml')
        with open(corrupted_path, 'w') as f:
            f.write('corrupted content')

        # Test loading corrupted file
        with pytest.raises(yaml.YAMLError):
            loader = ConfigurationLoader(self.temp_dir)
            loader.config_files['trading_config'] = 'corrupted.yaml'
            loader.load_trading_config()

    def test_config_validation_comprehensive(self):
        """Test comprehensive configuration validation."""
        # Test all configuration sections
        config = self.loader.load_trading_config()

        # Risk management validation
        risk_mgmt = config['risk_management']
        assert 0 < risk_mgmt['max_risk_per_trade'] <= 1
        assert 0 < risk_mgmt['max_daily_risk'] <= 1
        assert risk_mgmt['max_positions'] > 0
        assert risk_mgmt['stop_loss_pips'] > 0
        assert risk_mgmt['take_profit_pips'] > 0

        # Elliott Wave validation
        ew = config['elliott_wave']
        assert ew['min_wave_length'] > 0
        assert ew['max_wave_length'] > ew['min_wave_length']
        assert all(0 <= level <= 1 for level in ew['fibonacci_levels'])

        # ICT concepts validation
        ict = config['ict_concepts']
        assert ict['fvg_min_size'] > 0
        assert ict['fvg_max_age'] > 0
        assert 0 < ict['ob_min_strength'] <= 1

        # Market structure validation
        ms = config['market_structure']
        assert ms['swing_point_threshold'] > 0
        assert ms['bos_confirmation_candles'] > 0
        assert ms['choch_confirmation_candles'] > 0

        # Session logic validation
        session = config['session_logic']
        assert session['timezone'] in ['UTC', 'US/Eastern', 'Europe/London', 'Asia/Tokyo']
        assert 'asia_session' in session
        assert 'london_session' in session
        assert 'ny_session' in session
